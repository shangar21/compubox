import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class HitNet(nn.Module):
    def __init__(self, num_classes=1, weights=True):
        super(HitNet, self).__init__()

        # Load the pre-trained ResNet18 model
        resnet = models.resnet18(weights=weights)

        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add a custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



