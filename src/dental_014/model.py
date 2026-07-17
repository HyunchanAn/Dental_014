import torch
import torch.nn as nn
from torchvision import models

class OsteoporosisClassifier(nn.Module):
    """
    DenseNet121 based classifier for Osteoporosis risk screening.
    Outputs 3 classes: Normal (C1), Osteopenia (C2), Osteoporosis (C3).
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(OsteoporosisClassifier, self).__init__()
        
        # Load pre-trained DenseNet121
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)
        
        # Modify the classifier head with Dropout for regularization
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def get_model(num_classes=3, pretrained=True):
    """Factory function to get the model"""
    return OsteoporosisClassifier(num_classes=num_classes, pretrained=pretrained)
