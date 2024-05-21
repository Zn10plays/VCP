import torch
from PIL import Image

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2

preprocessor = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True), 
    v2.RandomResizedCrop((128 * 3, 128 * 2)),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize([.5,.5,.5],[.25,.25,.25])
])

import yaml

config = yaml.safe_load(open('constants/v1.yaml'))

class ImageDataset(Dataset):
    def __init__(self, maps, labels, image_path='./images/', transform=preprocessor):
        self.image_path = image_path
        self.transform = transform

        self.maps = pd.read_csv(maps)
        self.labels = pd.read_csv(labels)
        del self.labels['Unnamed: 0']

        # force order
        self.labels = self.labels[config['Dataset']['classes']]
        # remove redacted fields
        self.labels = self.labels.drop(columns=config['Dataset']['redactions'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.maps['file_name'][idx])
        image = Image.open(img_path).convert('RGB')

        label = self.labels.iloc[idx].values

        if self.transform:
            image = self.transform(image)

        return image.to('cuda'), torch.tensor(label, dtype=torch.float).to('cuda')


training_dataset = ImageDataset('data/train/features.csv', 'data/train/labels.csv', 'data/train/images/')

testing_dataset = ImageDataset('data/test/features.csv', 'data/test/labels.csv', 'data/test/images/')
