import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class SpectrogramsDataset(Dataset):

    def __init__(self, spec_dir, features_path, target='genre', transform=None):
        self.spec_dir = spec_dir
        self.features = pd.read_csv(features_path)
        self.target = target
        self.transform = transform

        #encode labels
        if target=='genre':
            encoder = LabelEncoder()
            self.features[target] = encoder.fit_transform(self.features[target])
            self.classes = encoder.classes_ #as torchvision.datasets.ImageFolder

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        row = self.features.iloc[i]
        spec_path = os.path.join(self.spec_dir, f"{row['id']}.png")
        with Image.open(spec_path) as im:
            #TODO remove? (to centralize the conversion in mp3_to_spec, but dunno if enough. eg. if we change to grayscale. but the actual dataset is RGBA so needed now)
            spec = im.convert('RGB') #as torchvision.datasets.ImageFolder
        if self.transform:
            spec = self.transform(spec)
        return spec, row[self.target]