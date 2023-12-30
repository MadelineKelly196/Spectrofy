import torch.nn.functional as F
from torchvision import models
from torch import nn


class DanceabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_arch = models.shufflenet_v2_x0_5() #https://pytorch.org/vision/stable/models.html
        self.fc = nn.Linear(1000, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.base_arch(x))
        x = F.sigmoid(self.fc(x))
        return x.squeeze(dim=-1)


class DanceabilityModelOld(nn.Module): #TODO delete if useless
    def __init__(self, channels, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 8, 5, padding='same')
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding='same')
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(16 * height//4 * width//4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))
        return x.squeeze(dim=-1)