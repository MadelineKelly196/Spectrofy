import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class SpectrogramsDataset(Dataset):

    def __init__(self, spec_dir, features_path, target='genre', transform=None):
        self.spec_dir = spec_dir
        self.features = pd.read_csv(features_path)
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        row = self.features.iloc[i]
        spec_path = os.path.join(self.spec_dir, f"{row['id']}.png")
        spec = Image.open(spec_path) #if ResourceWarning see torchvision.datasets.ImageFolder
        spec = spec.convert('RGB') #as torchvision.datasets.ImageFolder
        if self.transform:
            spec = self.transform(spec)
        label = row[self.target] #TODO if target=='genre' return class index instead (create self.classes as in ImageFolder)
        return spec, label