import torch.nn.functional as F
from torch import nn, flatten #TODO remove flatten if useless


class DanceabilityModel(nn.Module): #TODO debug
    def __init__(self, channels, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 8, 3, padding='same')
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(16 * height//4 * width//4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=-3) #no reshape() to work also without batch dim
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.sigmoid(self.fc2(self.dropout(x)))
        return x.squeeze(dim=-1)


class OldButWorking(nn.Module): #TODO delete after debugging
    def __init__(self, channels, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding='same')
        self.fc1 = nn.Linear(16 * 56 * 56, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, -3) #flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        return F.sigmoid(x)