import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

### Different NN Blocks
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

### Down block for U-Net 3+
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

class DownBlockBN(nn.Module):
    """Downsample by 2 then ConvBNReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

### Down block for deep models U-Net, U-Net++
class DownBlock(nn.Module):
    """Downsample by 2 then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

### Upblock common for all models
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


### Define simple U-Net architecture model

class UNetResNet34(nn.Module):
    """
    U-Net with ResNet34 encoder (ImageNet pretrained if requested).
    Input:  [B, 3, H, W] by default (see in_channels handling below)
    Output: [B, num_classes, H, W]
    """
    def __init__(self, num_classes: int, pretrained_backbone: bool = True, in_channels: int = 1):
        super().__init__()

        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        resnet = torchvision.models.resnet34(weights=weights)

        # --- Encoder (ResNet) ---
        # adapt first conv for grayscale if needed
        if in_channels == 3:
            self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained_backbone:
                # initialize from RGB weights by averaging channels
                with torch.no_grad():
                    conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
            self.enc0 = nn.Sequential(conv1, resnet.bn1, resnet.relu)

        self.pool = resnet.maxpool  # /4
        self.enc1 = resnet.layer1    # 64
        self.enc2 = resnet.layer2    # 128
        self.enc3 = resnet.layer3    # 256
        self.enc4 = resnet.layer4    # 512

        # --- Decoder (U-Net style) ---
        self.center = DoubleConv(512, 512)

        self.up3 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up2 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up1 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)
        # skip from enc0 is 64 channels; final up to full resolution
        self.up0 = UpBlock(in_ch=64,  skip_ch=64,  out_ch=64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)           # 64, /2
        x1 = self.enc1(self.pool(x0))  # 64, /4
        x2 = self.enc2(x1)          # 128, /8
        x3 = self.enc3(x2)          # 256, /16
        x4 = self.enc4(x3)          # 512, /32

        # Center
        y = self.center(x4)

        # Decoder
        y = self.up3(y, x3)         # /16
        y = self.up2(y, x2)         # /8
        y = self.up1(y, x1)         # /4
        y = self.up0(y, x0)         # /2 -> size of x0

        # Upsample to original input resolution
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(y)
        return logits


def get_unet_model(num_classes: int = 1, pretrained_backbone: bool = True, in_channels: int = 1):
    """
    UNetResNet34 (shallow U-Net, 5 encoder levels) with ImageNet-pretrained ResNet34 backbone.

    Architecture
    - Encoder: ResNet34 feature pyramid (down to /32 resolution).
    - Decoder: U-Net style upsampling with skip connections from encoder stages.
    - Output: per-pixel logits.

    Intended use
    - Binary semantic segmentation: set num_classes=1 and train with BCEWithLogitsLoss (optionally with pos_weight).
    - Multiclass exclusive segmentation: set num_classes=C>=2 and train with CrossEntropyLoss using target [B,H,W] long labels.

    Parameters
    ----------
    num_classes : int
        Number of output channels (classes) in the segmentation logits.
        - 1 for binary (foreground/background) with BCEWithLogitsLoss.
        - C for multiclass; use CrossEntropyLoss for exclusive classes.
    pretrained_backbone : bool
        If True, loads ImageNet pretrained weights for the ResNet34 encoder.
    in_channels : int
        Number of input image channels.
        - 1 for grayscale TEM.
        - 3 for RGB.

    Input/Output
    ------------
    Input:  torch.Tensor of shape [B, in_channels, H, W]
    Output: torch.Tensor of shape [B, num_classes, H, W] (logits)

    Notes
    -----
    - For in_channels=1 with pretrained_backbone=True, the first ResNet conv is initialized by averaging RGB weights.
    - Best performance is obtained when training and inference use identical preprocessing (tensor conversion + normalization).
    """
    return UNetResNet34(num_classes=num_classes, pretrained_backbone=pretrained_backbone, in_channels=in_channels)


### Define deep U-Net pretrained on ResNet50

### Common class for deep U-nets resnet encoder
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
        enc.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
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
        backbone="resnet50",
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


### Shallow U-Net ++ model 

class UNetPPResNet34(nn.Module):
    """
    U-Net++ (Nested U-Net) with a ResNet34 encoder (ImageNet-pretrained if requested).

    Output: logits [B, num_classes, H, W]
    Compatible with the same training pipeline (BCEWithLogits/Dice etc.).
    """
    def __init__(self, num_classes: int, pretrained_backbone: bool = True, in_channels: int = 1):
        super().__init__()

        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        resnet = torchvision.models.resnet34(weights=weights)

        # --- Encoder (ResNet34) ---
        if in_channels == 3:
            self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2, 64ch
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained_backbone:
                with torch.no_grad():
                    # average RGB weights -> grayscale init
                    conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
            self.enc0 = nn.Sequential(conv1, resnet.bn1, resnet.relu)

        self.pool = resnet.maxpool           # /4
        self.enc1 = resnet.layer1            # /4, 64ch
        self.enc2 = resnet.layer2            # /8, 128ch
        self.enc3 = resnet.layer3            # /16, 256ch
        self.enc4 = resnet.layer4            # /32, 512ch

        # Channel plan at each scale (x0..x4)
        ch0, ch1, ch2, ch3, ch4 = 64, 64, 128, 256, 512

        # --- U-Net++ nested decoder blocks ---
        # Notation: x_{i,j} where i=depth/scale (0..3), j=nested level (0..)
        # x_{i,0} are encoder features at that scale (after adjusting to correct H/W)
        # Each x_{i,j} is produced by combining:
        #   x_{i,0}, x_{i,1}, ..., x_{i,j-1} (same scale) and upsampled x_{i+1,j-1} (deeper scale)
        #
        # We use conv blocks after concatenation (standard U-Net++).

        # Level j=1
        self.x01 = DoubleConv(ch0 + ch1, ch0)                 # [x00, up(x10)]
        self.x11 = DoubleConv(ch1 + ch2, ch1)                 # [x10, up(x20)]
        self.x21 = DoubleConv(ch2 + ch3, ch2)                 # [x20, up(x30)]
        self.x31 = DoubleConv(ch3 + ch4, ch3)                 # [x30, up(x40)]

        # Level j=2
        self.x02 = DoubleConv(ch0 + ch0 + ch1, ch0)           # [x00, x01, up(x11)]
        self.x12 = DoubleConv(ch1 + ch1 + ch2, ch1)           # [x10, x11, up(x21)]
        self.x22 = DoubleConv(ch2 + ch2 + ch3, ch2)           # [x20, x21, up(x31)]

        # Level j=3
        self.x03 = DoubleConv(ch0 + ch0 + ch0 + ch1, ch0)     # [x00, x01, x02, up(x12)]
        self.x13 = DoubleConv(ch1 + ch1 + ch1 + ch2, ch1)     # [x10, x11, x12, up(x22)]

        # Level j=4
        self.x04 = DoubleConv(ch0 + ch0 + ch0 + ch0 + ch1, ch0)  # [x00,x01,x02,x03, up(x13)]

        # Final segmentation head (deep supervision optional; here we use x04 only)
        self.head = nn.Conv2d(ch0, num_classes, kernel_size=1)

    @staticmethod
    def _up_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # --- Encoder forward ---
        x00 = self.enc0(x)               # 64, /2
        x10 = self.enc1(self.pool(x00))  # 64, /4
        x20 = self.enc2(x10)             # 128, /8
        x30 = self.enc3(x20)             # 256, /16
        x40 = self.enc4(x30)             # 512, /32

        # --- U-Net++ nested decoder ---
        x01 = self.x01(torch.cat([x00, self._up_to(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up_to(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self._up_to(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, self._up_to(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self._up_to(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up_to(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self._up_to(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self._up_to(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self._up_to(x22, x10)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self._up_to(x13, x00)], dim=1))

        # --- Output logits at original resolution ---
        y = self._up_to(x04, x)          # from /2 back to full H,W
        logits = self.head(y)
        return logits


def get_unetpp_model(num_classes: int, pretrained_backbone: bool = True, in_channels: int = 1):
    """
    UNetPPResNet34 (U-Net++ / Nested U-Net, 5 encoder levels) with ImageNet-pretrained ResNet34 backbone.

    Architecture
    - Encoder: ResNet34 (down to /32).
    - Decoder: U-Net++ nested skip connections (dense refinement of skip pathways).
    - Output: per-pixel logits from the most refined decoder node (no deep supervision head by default).

    Intended use
    - Same task conventions as UNetResNet34.
    - Often improves boundary refinement vs plain U-Net at the cost of higher compute/memory.

    Parameters
    ----------
    num_classes : int
        Number of output channels/classes in logits.
    pretrained_backbone : bool
        If True, loads ImageNet pretrained weights for the ResNet34 encoder.
    in_channels : int
        Number of input channels (1 for grayscale TEM, 3 for RGB).

    Input/Output
    ------------
    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W] logits

    Notes
    -----
    - Significantly heavier than plain U-Net due to additional nested decoder blocks and concatenations.
    - For unstable training on imbalanced data, prefer starting with BCEWithLogitsLoss before adding overlap losses.
    """
    return UNetPPResNet34(num_classes=num_classes, pretrained_backbone=pretrained_backbone, in_channels=in_channels)


### Make U-Net++ deep

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


### Make shallow U-Net3+ Model
class UNet3PlusResNet34(nn.Module):
    """
    UNet3PlusResNet34 (UNet3+ / full-scale skip fusion) with ResNet34 encoder (5 levels).

    Architecture
    - Encoder: ResNet34 features at multiple scales (down to /32).
    - Decoder: full-scale fusion at each decoder stage: each stage aggregates features from ALL encoder scales
    (upsampled/downsampled to the target resolution) plus the previous decoder feature.
    - Output: per-pixel logits.

    Intended use
    - Designed to improve multi-scale feature fusion vs standard U-Net/UNet++.
    - Useful when object sizes vary and you need both fine detail and global context at each stage.

    Parameters
    ----------
    in_channels : int
        Input channels (1 grayscale TEM, 3 RGB).
    num_classes : int
        Output channels/classes in logits.
    pretrained : bool
        If True, loads ImageNet pretrained weights for ResNet34.
    fuse_ch : int
        Channel width used to project each encoder feature map before fusion. Larger fuse_ch increases capacity and cost.
    up_mode : str
        Upsampling interpolation mode, typically "bilinear".

    Input/Output
    ------------
    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W] logits

    Notes
    -----
    - Less nested than UNet++ but performs wide multi-scale concatenations; memory usage can still be significant.
    - H and W divisible by 32 are preferred due to the /32 encoder level.
    """
    def __init__(self, in_channels=1, num_classes=1, pretrained=True, fuse_ch=64, up_mode="bilinear"):
        super().__init__()
        self.up_mode = up_mode

        w = models.ResNet34_Weights.DEFAULT if pretrained else None
        enc = models.resnet34(weights=w)

        # adapt conv1 to grayscale if needed
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

        self.conv1 = enc.conv1
        self.bn1 = enc.bn1
        self.relu = enc.relu
        self.maxpool = enc.maxpool
        self.layer1 = enc.layer1
        self.layer2 = enc.layer2
        self.layer3 = enc.layer3
        self.layer4 = enc.layer4

        # Encoder channels for ResNet34 at each level
        c0, c1, c2, c3, c4 = 64, 64, 128, 256, 512

        # Per-level projection to fuse_ch (UNet3+ unifies channels before concat)
        self.proj0 = ConvBNReLU(c0, fuse_ch, k=3, p=1)
        self.proj1 = ConvBNReLU(c1, fuse_ch, k=3, p=1)
        self.proj2 = ConvBNReLU(c2, fuse_ch, k=3, p=1)
        self.proj3 = ConvBNReLU(c3, fuse_ch, k=3, p=1)
        self.proj4 = ConvBNReLU(c4, fuse_ch, k=3, p=1)

        # Decoder projections (for the "previous deeper decoder" signal)
        self.projd3 = ConvBNReLU(fuse_ch * 5, fuse_ch, k=3, p=1)  # after fusion at /16
        self.projd2 = ConvBNReLU(fuse_ch * 6, fuse_ch, k=3, p=1)  # after fusion at /8
        self.projd1 = ConvBNReLU(fuse_ch * 6, fuse_ch, k=3, p=1)  # after fusion at /4
        self.projd0 = ConvBNReLU(fuse_ch * 6, fuse_ch, k=3, p=1)  # after fusion at /2

        # Output head
        self.out_conv = nn.Conv2d(fuse_ch, num_classes, kernel_size=1)

    def _resize_to(self, x, ref):
        return F.interpolate(
            x, size=ref.shape[-2:], mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None
        )

    def forward(self, x):
        # ----- Encoder -----
        e0 = self.relu(self.bn1(self.conv1(x)))     # /2, 64
        e1 = self.layer1(self.maxpool(e0))          # /4, 64
        e2 = self.layer2(e1)                        # /8, 128
        e3 = self.layer3(e2)                        # /16, 256
        e4 = self.layer4(e3)                        # /32, 512

        # Project encoder features to common channel width
        p0 = self.proj0(e0)  # /2
        p1 = self.proj1(e1)  # /4
        p2 = self.proj2(e2)  # /8
        p3 = self.proj3(e3)  # /16
        p4 = self.proj4(e4)  # /32

        # ----- Decoder d3 at /16 (fuse all encoder scales) -----
        # target ref = p3
        f3 = torch.cat([
            self._resize_to(p0, p3),
            self._resize_to(p1, p3),
            self._resize_to(p2, p3),
            p3,
            self._resize_to(p4, p3),
        ], dim=1)  # 5*fuse_ch
        d3 = self.projd3(f3)  # /16, fuse_ch

        # ----- Decoder d2 at /8 (fuse all enc + d3) -----
        # target ref = p2
        f2 = torch.cat([
            self._resize_to(p0, p2),
            self._resize_to(p1, p2),
            p2,
            self._resize_to(p3, p2),
            self._resize_to(p4, p2),
            self._resize_to(d3, p2),
        ], dim=1)  # 6*fuse_ch
        d2 = self.projd2(f2)  # /8

        # ----- Decoder d1 at /4 (fuse all enc + d2) -----
        # target ref = p1
        f1 = torch.cat([
            self._resize_to(p0, p1),
            p1,
            self._resize_to(p2, p1),
            self._resize_to(p3, p1),
            self._resize_to(p4, p1),
            self._resize_to(d2, p1),
        ], dim=1)  # 6*fuse_ch
        d1 = self.projd1(f1)  # /4

        # ----- Decoder d0 at /2 (fuse all enc + d1) -----
        # target ref = p0
        f0 = torch.cat([
            p0,
            self._resize_to(p1, p0),
            self._resize_to(p2, p0),
            self._resize_to(p3, p0),
            self._resize_to(p4, p0),
            self._resize_to(d1, p0),
        ], dim=1)  # 6*fuse_ch
        d0 = self.projd0(f0)  # /2

        # ----- Output to /1 -----
        d0_up = F.interpolate(
            d0, scale_factor=2, mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None
        )  # /1
        logits = self.out_conv(d0_up)  # [B, num_classes, H, W]
        return logits


### Deep U-Net3+

class UNet3PlusResNet50Deep(nn.Module):
    """
    UNet3PlusResNet50Deep (UNet3+ with 7 levels): ResNet50 encoder + 2 extra down levels + full-scale fusion.

    Architecture
    - Encoder levels:
    e0: /2, e1: /4, e2: /8, e3: /16, e4: /32 (ResNet50)
    e5: /64, e6: /128 (extra down blocks)
    - Decoder stages:
    d5: /64, d4: /32, d3: /16, d2: /8, d1: /4, d0: /2
    Each stage fuses all encoder scales + previous decoder feature at the target resolution.

    Intended use
    - Strong multi-scale aggregation with larger receptive field, suitable for heterogeneous magnifications.
    - Heavier and more sensitive to optimization settings than shallow U-Net.

    Parameters
    ----------
    in_channels : int
        Input channels (1 grayscale TEM, 3 RGB).
    num_classes : int
        Output channels/classes in logits.
    pretrained : bool
        If True, loads ImageNet pretrained weights for ResNet50.
    fuse_ch : int
        Projection width for each scale before fusion. Controls compute/memory strongly.
    extra_down_channels : tuple[int, int]
        Channels for extra down levels (e5,e6).
    up_mode : str
        Upsampling mode ("bilinear" typical).

    Input/Output
    ------------
    Input:  [B, in_channels, H, W]
    Output: [B, num_classes, H, W] logits

    Notes
    -----
    - Prefer patch sizes divisible by 128 to reduce interpolation/boundary artifacts.
    - If the model collapses to all-positive/all-negative, reduce overlap-loss weight, lower LR, and use gradient clipping.
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        pretrained=True,
        fuse_ch=64,
        extra_down_channels=(1024, 1024),  # e5,e6 widths after layer4
        up_mode="bilinear",
    ):
        super().__init__()
        self.up_mode = up_mode

        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        enc = models.resnet50(weights=w)

        # adapt conv1 to grayscale if needed
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

        self.conv1 = enc.conv1
        self.bn1 = enc.bn1
        self.relu = enc.relu
        self.maxpool = enc.maxpool
        self.layer1 = enc.layer1
        self.layer2 = enc.layer2
        self.layer3 = enc.layer3
        self.layer4 = enc.layer4

        # ResNet50 encoder channels at each level
        c0, c1, c2, c3, c4 = 64, 256, 512, 1024, 2048

        e5_ch, e6_ch = extra_down_channels
        self.down5 = DownBlockBN(c4, e5_ch)   # /64
        self.down6 = DownBlockBN(e5_ch, e6_ch) # /128

        # project all encoder features to fuse_ch
        self.proj0 = ConvBNReLU(c0, fuse_ch)
        self.proj1 = ConvBNReLU(c1, fuse_ch)
        self.proj2 = ConvBNReLU(c2, fuse_ch)
        self.proj3 = ConvBNReLU(c3, fuse_ch)
        self.proj4 = ConvBNReLU(c4, fuse_ch)
        self.proj5 = ConvBNReLU(e5_ch, fuse_ch)
        self.proj6 = ConvBNReLU(e6_ch, fuse_ch)

        # decoder fusion convs
        # d5 (/64): fuse encoder 0..6 => 7*fuse_ch
        self.fuse5 = ConvBNReLU(7 * fuse_ch, fuse_ch)
        # d4 (/32): fuse encoder 0..6 + d5 => 8*fuse_ch
        self.fuse4 = ConvBNReLU(8 * fuse_ch, fuse_ch)
        # d3 (/16): fuse encoder 0..6 + d4 => 8*fuse_ch
        self.fuse3 = ConvBNReLU(8 * fuse_ch, fuse_ch)
        # d2 (/8): fuse encoder 0..6 + d3 => 8*fuse_ch
        self.fuse2 = ConvBNReLU(8 * fuse_ch, fuse_ch)
        # d1 (/4): fuse encoder 0..6 + d2 => 8*fuse_ch
        self.fuse1 = ConvBNReLU(8 * fuse_ch, fuse_ch)
        # d0 (/2): fuse encoder 0..6 + d1 => 8*fuse_ch
        self.fuse0 = ConvBNReLU(8 * fuse_ch, fuse_ch)

        self.out_conv = nn.Conv2d(fuse_ch, num_classes, kernel_size=1)

    def _resize_to(self, x, ref):
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None,
        )

    def forward(self, x):
        # ----- encoder -----
        e0 = self.relu(self.bn1(self.conv1(x)))       # /2, 64
        e1 = self.layer1(self.maxpool(e0))            # /4, 256
        e2 = self.layer2(e1)                          # /8, 512
        e3 = self.layer3(e2)                          # /16, 1024
        e4 = self.layer4(e3)                          # /32, 2048
        e5 = self.down5(e4)                           # /64
        e6 = self.down6(e5)                           # /128

        # project to fuse_ch
        p0 = self.proj0(e0)  # /2
        p1 = self.proj1(e1)  # /4
        p2 = self.proj2(e2)  # /8
        p3 = self.proj3(e3)  # /16
        p4 = self.proj4(e4)  # /32
        p5 = self.proj5(e5)  # /64
        p6 = self.proj6(e6)  # /128

        # ----- decoder d5 at /64: fuse all encoder scales -----
        ref = p5
        f5 = torch.cat([
            self._resize_to(p0, ref),
            self._resize_to(p1, ref),
            self._resize_to(p2, ref),
            self._resize_to(p3, ref),
            self._resize_to(p4, ref),
            p5,
            self._resize_to(p6, ref),
        ], dim=1)
        d5 = self.fuse5(f5)  # /64

        # ----- decoder d4 at /32: fuse all enc + d5 -----
        ref = p4
        f4 = torch.cat([
            self._resize_to(p0, ref),
            self._resize_to(p1, ref),
            self._resize_to(p2, ref),
            self._resize_to(p3, ref),
            p4,
            self._resize_to(p5, ref),
            self._resize_to(p6, ref),
            self._resize_to(d5, ref),
        ], dim=1)
        d4 = self.fuse4(f4)  # /32

        # ----- decoder d3 at /16 -----
        ref = p3
        f3 = torch.cat([
            self._resize_to(p0, ref),
            self._resize_to(p1, ref),
            self._resize_to(p2, ref),
            p3,
            self._resize_to(p4, ref),
            self._resize_to(p5, ref),
            self._resize_to(p6, ref),
            self._resize_to(d4, ref),
        ], dim=1)
        d3 = self.fuse3(f3)  # /16

        # ----- decoder d2 at /8 -----
        ref = p2
        f2 = torch.cat([
            self._resize_to(p0, ref),
            self._resize_to(p1, ref),
            p2,
            self._resize_to(p3, ref),
            self._resize_to(p4, ref),
            self._resize_to(p5, ref),
            self._resize_to(p6, ref),
            self._resize_to(d3, ref),
        ], dim=1)
        d2 = self.fuse2(f2)  # /8

        # ----- decoder d1 at /4 -----
        ref = p1
        f1 = torch.cat([
            self._resize_to(p0, ref),
            p1,
            self._resize_to(p2, ref),
            self._resize_to(p3, ref),
            self._resize_to(p4, ref),
            self._resize_to(p5, ref),
            self._resize_to(p6, ref),
            self._resize_to(d2, ref),
        ], dim=1)
        d1 = self.fuse1(f1)  # /4

        # ----- decoder d0 at /2 -----
        ref = p0
        f0 = torch.cat([
            p0,
            self._resize_to(p1, ref),
            self._resize_to(p2, ref),
            self._resize_to(p3, ref),
            self._resize_to(p4, ref),
            self._resize_to(p5, ref),
            self._resize_to(p6, ref),
            self._resize_to(d1, ref),
        ], dim=1)
        d0 = self.fuse0(f0)  # /2

        # ----- output to /1 -----
        d0_up = F.interpolate(
            d0, scale_factor=2, mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None
        )
        logits = self.out_conv(d0_up)  # [B, num_classes, H, W]
        return logits
