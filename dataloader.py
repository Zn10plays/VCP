import torch
from PIL import Image

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import yaml

config = yaml.full_load(open('constants/v2.yaml'))

conditional_preprocesses = v2.Compose([
    v2.RandomChoice([
        v2.RandomResizedCrop(size=config['ViT']['image_size']),
        v2.Resize(size=config['ViT']['image_size'])
    ], [.9, .1]),
    v2.RandomApply([v2.RandomRotation([-180, 180])], .2),
    v2.RandomHorizontalFlip(.3),
    v2.RandomVerticalFlip(.3),
])


def preprocessor(images, random_crop=True, device='cuda'):
    images = v2.functional.to_image(images)
    images = v2.functional.to_dtype(images, torch.uint8)
    if random_crop:
        images = conditional_preprocesses(images)

    images = v2.functional.to_dtype(images, torch.float32, scale=True)

    images = 2 * images - 1

    return images.to(device)

class ImageDataset(Dataset):
    def __init__(self, maps, labels, image_path='./images/', transform=preprocessor, augmentation: bool = False):
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

        return image.to('cuda'), torch.tensor(label, dtype=torch.float).to('cuda')


training_dataset = ImageDataset('data/train/features.csv',
                                'data/train/labels.csv',
                                'data/train/images/',
                                augmentation=True)

testing_dataset = ImageDataset('data/test/features.csv',
                               'data/test/labels.csv',
                               'data/test/images/',
                               augmentation=False)
