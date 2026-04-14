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


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    """Downsample by 2 then ConvBNReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


def _make_resnet_encoder(backbone: str, in_channels: int, pretrained: bool):
    if backbone == "resnet34":
        w = models.ResNet34_Weights.DEFAULT if pretrained else None
        enc = models.resnet34(weights=w)
        enc_ch = (64, 64, 128, 256, 512)
    elif backbone == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        enc = models.resnet50(weights=w)
        enc_ch = (64, 256, 512, 1024, 2048)
    else:
        raise ValueError("backbone must be 'resnet34' or 'resnet50'")

    if in_channels != 3:
        old = enc.conv1
        enc.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if pretrained and in_channels == 1:
            with torch.no_grad():
                enc.conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True) / 3.0)

    return enc, enc_ch


class UNetPPResNetDeep(nn.Module):
    """
    UNetPPResNetDeep (7-level U-Net++): ResNet encoder + 2 extra down levels + nested decoder.

    Architecture
    - Same 7 encoder resolution levels as UNetResNetDeep.
    - Decoder uses UNet++ nested dense skip connections across levels and refinement depths.
    - Output produced from the most refined node at the highest decoder resolution.

    Intended use
    - Semantic segmentation with stronger multi-stage refinement (often better boundaries),
    at significantly higher compute/memory cost.

    Parameters
    ----------
    backbone : str
        "resnet34" or "resnet50".
    in_channels : int
        Input channels.
    num_classes : int
        Output channels/classes in logits.
    pretrained : bool
        If True, loads ImageNet pretrained weights for the backbone.
    extra_down_channels : tuple[int, int]
        Channel widths for extra down levels (e5,e6).
    dec_channels : tuple[int, ...]
        Decoder channel widths per level (levels 0..6). Must have length 7.
    up_mode : str
        Upsampling mode ("bilinear" typical).

    Input/Output
    ------------
    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W] logits

    Notes
    -----
    - Very heavy; expect 2–3× slower than plain U-Net on the same input size.
    - Prefer smaller patch sizes and strong class-imbalance handling (sampling and/or loss) for TEM nanoparticles.
    """
    def __init__(
        self,
        backbone: str = "resnet34",
        in_channels: int = 1,
        num_classes: int = 1,
        pretrained: bool = True,
        extra_down_channels=(512, 512),  # e5, e6; for resnet50 you may want (1024,1024)
        dec_channels=(64, 64, 128, 256, 512, 512, 512),
        up_mode: str = "bilinear",
    ):
        super().__init__()
        self.up_mode = up_mode

        enc, enc_ch_0_4 = _make_resnet_encoder(backbone, in_channels, pretrained)
        self.encoder = enc

        # ResNet stem and blocks
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # level0 (/2)
        self.maxpool = enc.maxpool                               # /4
        self.layer1 = enc.layer1                                 # level1
        self.layer2 = enc.layer2                                 # level2
        self.layer3 = enc.layer3                                 # level3
        self.layer4 = enc.layer4                                 # level4

        c0, c1, c2, c3, c4 = enc_ch_0_4
        e5, e6 = extra_down_channels

        # two extra down levels
        self.down5 = DownBlock(c4, e5)
        self.down6 = DownBlock(e5, e6)

        # encoder channel list for levels 0..6
        enc_ch = (c0, c1, c2, c3, c4, e5, e6)
        dec_ch = tuple(dec_channels)
        assert len(dec_ch) == 7, "dec_channels must have 7 entries (levels 0..6)."

        # map encoder features to decoder-width at x{i}_0
        self.x0_0 = DoubleConv(enc_ch[0], dec_ch[0])
        self.x1_0 = DoubleConv(enc_ch[1], dec_ch[1])
        self.x2_0 = DoubleConv(enc_ch[2], dec_ch[2])
        self.x3_0 = DoubleConv(enc_ch[3], dec_ch[3])
        self.x4_0 = DoubleConv(enc_ch[4], dec_ch[4])
        self.x5_0 = DoubleConv(enc_ch[5], dec_ch[5])
        self.x6_0 = DoubleConv(enc_ch[6], dec_ch[6])

        # Nested conv blocks x{i}_{j}, for i=0..5, j=1..(6-i)
        # Input to x{i}_{j}: concat( x{i}_0..x{i}_{j-1}, up(x{i+1}_{j-1}) )
        # Channels: j*dec_ch[i] + dec_ch[i+1]
        blocks = {}
        for i in range(0, 6):
            for j in range(1, 7 - i):
                in_ch = j * dec_ch[i] + dec_ch[i + 1]
                out_ch = dec_ch[i]
                blocks[f"x{i}_{j}"] = DoubleConv(in_ch, out_ch)
        self.blocks = nn.ModuleDict(blocks)

        # final up to full res (/1) + 1x1 head
        self.out_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False if up_mode == "bilinear" else None),
            DoubleConv(dec_ch[0], dec_ch[0]),
            nn.Conv2d(dec_ch[0], num_classes, kernel_size=1),
        )

    def _up_to(self, x, ref):
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None,
        )

    def forward(self, x):
        # encoder feature extraction
        e0 = self.stem(x)               # /2
        e1 = self.layer1(self.maxpool(e0))  # /4
        e2 = self.layer2(e1)            # /8
        e3 = self.layer3(e2)            # /16
        e4 = self.layer4(e3)            # /32
        e5 = self.down5(e4)             # /64
        e6 = self.down6(e5)             # /128

        # project to decoder width (x{i}_0)
        x0_0 = self.x0_0(e0)
        x1_0 = self.x1_0(e1)
        x2_0 = self.x2_0(e2)
        x3_0 = self.x3_0(e3)
        x4_0 = self.x4_0(e4)
        x5_0 = self.x5_0(e5)
        x6_0 = self.x6_0(e6)

        # store nodes in a dict for nested construction
        X = {(0,0): x0_0, (1,0): x1_0, (2,0): x2_0, (3,0): x3_0, (4,0): x4_0, (5,0): x5_0, (6,0): x6_0}

        # build UNet++ dense skip nodes
        for j in range(1, 7):                # nesting depth
            for i in range(0, 7 - j):        # level
                ref = X[(i, 0)]
                up = self._up_to(X[(i + 1, j - 1)], ref)
                to_cat = [X[(i, k)] for k in range(0, j)] + [up]
                X[(i, j)] = self.blocks[f"x{i}_{j}"](torch.cat(to_cat, dim=1))

        logits = self.out_up(X[(0, 6)])      # x0_6 is /2 -> up to /1
        return logits