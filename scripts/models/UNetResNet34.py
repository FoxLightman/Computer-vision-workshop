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


class UNetResNet34(nn.Module):
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