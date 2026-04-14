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


class UNetPPResNet34(nn.Module):
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
    return UNetPPResNet34(num_classes=num_classes, pretrained_backbone=pretrained_backbone, in_channels=in_channels)