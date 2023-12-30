import torch.nn.functional as F
from torch import nn


class DanceabilityModel(nn.Module):
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=-3) #no reshape() to work also without batch dim
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.sigmoid(self.fc3(self.dropout(x)))
        return x.squeeze(dim=-1)