import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    """Downsample by 2 then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample by 2, concat skip, then DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode=self.mode, align_corners=False if self.mode=="bilinear" else None)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


def _make_resnet_encoder(backbone: str, in_channels: int, pretrained: bool):
    if backbone == "resnet34":
        w = models.ResNet34_Weights.DEFAULT if pretrained else None
        enc = models.resnet34(weights=w)
        chs = (64, 64, 128, 256, 512)   # c0..c4
    elif backbone == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        enc = models.resnet50(weights=w)
        chs = (64, 256, 512, 1024, 2048)
    else:
        raise ValueError("backbone must be 'resnet34' or 'resnet50'")

    # adapt first conv to in_channels
    if in_channels != 3:
        old = enc.conv1
        enc.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, bias=False)
        if pretrained and in_channels == 1:
            with torch.no_grad():
                enc.conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True) / 3.0)

    return enc, chs


class UNetResNetDeep(nn.Module):
    """
    UNetResNetDeep (7-level U-Net): ResNet encoder + 2 extra downsampling levels.

    Architecture
    - Encoder levels:
    e0: /2  (stem conv)
    e1: /4
    e2: /8
    e3: /16
    e4: /32  (ResNet deepest feature)
    e5: /64  (extra down block)
    e6: /128 (extra down block)
    - Decoder mirrors these levels back to full resolution.

    Intended use
    - Same loss conventions as other semantic segmentation models in this library.
    - Useful when larger receptive field/context is required.

    Parameters
    ----------
    backbone : str
        Encoder backbone identifier: "resnet34" or "resnet50".
    in_channels : int
        Input channels (1 grayscale, 3 RGB).
    num_classes : int
        Output channels/classes in logits.
    pretrained : bool
        If True, loads ImageNet pretrained weights for the ResNet backbone.
    extra_down_channels : tuple[int, int]
        Channel widths for the two extra down levels (e5, e6), e.g. (1024,1024).
    up_mode : str
        Upsampling interpolation mode, typically "bilinear".

    Input/Output
    ------------
    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W] logits

    Notes
    -----
    - Because the network downsamples by a factor of 128, H and W divisible by 128 reduce boundary artifacts.
    - This model is substantially heavier than 5-level U-Net, especially with ResNet50 backbone.
    """
    def __init__(
        self,
        backbone="resnet34",
        in_channels=1,
        num_classes=1,
        pretrained=True,
        extra_down_channels=(1024, 1024),   # two extra levels after resnet deepest
        up_mode="bilinear",
    ):
        super().__init__()
        self.encoder, (c0, c1, c2, c3, c4) = _make_resnet_encoder(backbone, in_channels, pretrained)

        # ResNet "stem" pieces
        self.stem = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )  # output: c0 channels, stride 2
        self.maxpool = self.encoder.maxpool  # /4

        # ResNet layers (skip features)
        self.layer1 = self.encoder.layer1  # c1, /4
        self.layer2 = self.encoder.layer2  # c2, /8
        self.layer3 = self.encoder.layer3  # c3, /16
        self.layer4 = self.encoder.layer4  # c4, /32

        # Two extra down levels: /64 and /128
        e5, e6 = extra_down_channels
        self.down5 = DownBlock(c4, e5)
        self.down6 = DownBlock(e5, e6)

        # Decoder (mirror: 6 ups back to /1)
        self.up6 = UpBlock(in_ch=e6, skip_ch=e5, out_ch=e5, mode=up_mode)  # /128 -> /64
        self.up5 = UpBlock(in_ch=e5, skip_ch=c4, out_ch=512, mode=up_mode) # /64  -> /32
        self.up4 = UpBlock(in_ch=512, skip_ch=c3, out_ch=256, mode=up_mode) # /32 -> /16
        self.up3 = UpBlock(in_ch=256, skip_ch=c2, out_ch=128, mode=up_mode) # /16 -> /8
        self.up2 = UpBlock(in_ch=128, skip_ch=c1, out_ch=64,  mode=up_mode) # /8  -> /4
        self.up1 = UpBlock(in_ch=64,  skip_ch=c0, out_ch=64,  mode=up_mode) # /4  -> /2

        # Final up to /1 (undo the stem stride 2)
        self.out_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False if up_mode=="bilinear" else None),
            DoubleConv(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)               # c0, /2
        x1 = self.maxpool(s0)           # /4
        s1 = self.layer1(x1)            # c1, /4
        s2 = self.layer2(s1)            # c2, /8
        s3 = self.layer3(s2)            # c3, /16
        s4 = self.layer4(s3)            # c4, /32

        s5 = self.down5(s4)             # e5, /64
        s6 = self.down6(s5)             # e6, /128

        # Decoder
        x = self.up6(s6, s5)            # /64
        x = self.up5(x,  s4)            # /32
        x = self.up4(x,  s3)            # /16
        x = self.up3(x,  s2)            # /8
        x = self.up2(x,  s1)            # /4
        x = self.up1(x,  s0)            # /2

        logits = self.out_up(x)         # /1
        return logits