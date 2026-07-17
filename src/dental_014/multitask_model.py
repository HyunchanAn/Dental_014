import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class OsteoMultiTaskNet(nn.Module):
    """
    End-to-End Multi-task Architecture for Osteoporosis (Mask-free Version).
    Head 1: Image Reconstruction (Autoencoder style) to regularize backbone without Ground Truth masks.
    Head 2: Osteoporosis Classification (C1, C2, C3)
    """
    def __init__(self, in_channels=3, out_channels=3, num_classes=3, features=[64, 128, 256, 512]):
        super(OsteoMultiTaskNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part (Shared Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # e.g. 512 -> 1024
        
        # Up part (Head 1: Reconstruction Decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Head 2: Classification (Uses Bottleneck Features)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(features[-1]*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        bottleneck_features = self.bottleneck(x)
        
        # ----- Head 2: Classification -----
        pooled = self.global_avg_pool(bottleneck_features)
        pooled = torch.flatten(pooled, 1)
        class_logits = self.classifier(pooled)
        
        # ----- Head 1: Image Reconstruction -----
        skip_connections = skip_connections[::-1]
        x_up = bottleneck_features
        for idx in range(0, len(self.ups), 2):
            x_up = self.ups[idx](x_up)
            skip_connection = skip_connections[idx//2]
            
            if x_up.shape != skip_connection.shape:
                x_up = TF.resize(x_up, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x_up), dim=1)
            x_up = self.ups[idx+1](concat_skip)
            
        recon_logits = self.final_conv(x_up)
        
        return recon_logits, class_logits
