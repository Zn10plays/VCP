import torch
from PIL import Image

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import yaml

preprocessor = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8),
])

postprocessor = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Normalize([.61862556, .57236481, .57478806],[.31973445, .32038794, .31461327])
])

config = yaml.safe_load(open('constants/v2.yaml'))


class ImageDataset(Dataset):
    def __init__(self, maps, labels, image_path='./images/', transform: bool = True, augmentation: bool = False):
        self.image_path = image_path
        self.transform = transform

        self.maps = pd.read_csv(maps)
        self.labels = pd.read_csv(labels)

        self.augmentation = augmentation

        # force order
        self.labels = self.labels[config['Dataset']['classes']]
        # remove redacted fields
        self.labels = self.labels.drop(columns=config['Dataset']['redactions'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.maps['cover_filename'][idx])
        image = Image.open(img_path).convert('RGB')

        label = self.labels.iloc[idx].values

        if self.transform:
            image = preprocessor(image)

        if self.augmentation:
            image = v2.functional.resized_crop(image, config['Model']['image_size'])

        if self.transform:
            image = postprocessor(image)


        return image.to('cuda'), torch.tensor(label, dtype=torch.float).to('cuda')


training_dataset = ImageDataset('data/train/features.csv', 'data/train/labels.csv', 'data/train/images/', augmentation=True)

testing_dataset = ImageDataset('data/test/features.csv', 'data/test/labels.csv', 'data/test/images/', transform=True, augmentation=False)
