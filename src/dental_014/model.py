import os
import torch
import torch.nn as nn
from torchvision import models
from dental_014.unet_model import UNet


class MaskAttentionResNet(nn.Module):
    """
    Mask-Attention ResNet50 for Osteoporosis risk screening.
    Uses U-Net to generate a mask from the input image,
    and appends it as a 4th channel to ResNet50.
    """

    def __init__(self, num_classes=3, pretrained=True, unet_weights_path=None):
        super(MaskAttentionResNet, self).__init__()

        # 1. Initialize U-Net
        self.unet = UNet(in_channels=3, out_channels=1)
        if unet_weights_path and os.path.exists(unet_weights_path):
            self.unet.load_state_dict(
                torch.load(unet_weights_path, map_location="cpu", weights_only=True)
            )
            print(f"Loaded U-Net weights from {unet_weights_path}")

        # Freeze U-Net
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        # 2. Initialize ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Modify the first convolution to accept 4 channels
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize the weights of the new conv layer
        if pretrained:
            with torch.no_grad():
                # Copy the first 3 channels
                self.resnet.conv1.weight[:, :3, :, :] = original_conv.weight
                # Initialize the 4th channel with the mean of the RGB weights
                self.resnet.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)

        # Modify the classifier head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x is (B, 3, H, W)
        with torch.no_grad():
            self.unet.eval()  # Ensure it's in eval mode
            mask_logits = self.unet(x)
            mask = torch.sigmoid(mask_logits)  # (B, 1, H, W)

        # Concatenate x and mask -> (B, 4, H, W)
        x_4ch = torch.cat([x, mask], dim=1)

        return self.resnet(x_4ch)


def get_model(num_classes=3, pretrained=True, unet_weights_path=None):
    """Factory function to get the model"""
    return MaskAttentionResNet(
        num_classes=num_classes,
        pretrained=pretrained,
        unet_weights_path=unet_weights_path,
    )
