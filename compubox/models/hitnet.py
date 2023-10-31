import torch.nn as nn
import torch.nn.functional as F
import torch

class HitNet(nn.Module):
    def __init__(self, img_size=(480, 480)):
        super().__init__()
        self.image_size = img_size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 1, 5)
        flat_params = ((img_size[0] - 4) // 2, (img_size[1] - 4) // 2)
        flat_params = ((flat_params[0] - 4) // 2, (flat_params[1] - 4) // 2)
        self.fc1 = nn.Linear(flat_params[0] * flat_params[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

