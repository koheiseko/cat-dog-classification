import torch.nn as nn
from torchvision import models 

class ResNetTransfer(nn.Module):
    def __init__(self, n_classes:int):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)

        for param in self.resnet.parameters():
            param.requires_grad = False

        in_features_original = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(in_features_original, n_classes)

    def forward(self, x):
        return self.resnet(x)