import torch.nn.functional as F
from torchvision import models
from torch import nn


class DanceabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_arch = models.squeezenet1_1() #https://pytorch.org/vision/stable/models.html
        self.fc = nn.Linear(1000, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.base_arch(x))
        x = F.sigmoid(self.fc(x))
        return x.squeeze(dim=-1)