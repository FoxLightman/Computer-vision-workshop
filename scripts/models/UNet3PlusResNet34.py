import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

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