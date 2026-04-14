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


class DownBlock(nn.Module):
    """Downsample by 2 then ConvBNReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UNet3PlusResNet50Deep(nn.Module):
    """
    UNet3+ with ResNet50 encoder + 2 extra down levels => 7 levels total.

    Encoder feature scales:
      e0: /2    (stem conv1)
      e1: /4    (layer1)
      e2: /8    (layer2)
      e3: /16   (layer3)
      e4: /32   (layer4)
      e5: /64   (extra down)
      e6: /128  (extra down)

    Decoder stages built at:
      d5: /64
      d4: /32
      d3: /16
      d2: /8
      d1: /4
      d0: /2
    Output upsample to /1 and 1x1 head => logits [B, num_classes, H, W]

    Compatible with your training pipeline (BCEWithLogitsLoss / focal / etc.).
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
        self.down5 = DownBlock(c4, e5_ch)   # /64
        self.down6 = DownBlock(e5_ch, e6_ch) # /128

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