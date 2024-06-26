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


@torch.no_grad()
def preprocessor(images, random_crop=True, vae_prep=False):
    images = v2.functional.to_image(images)
    images = v2.functional.to_dtype(images, torch.uint8)
    if random_crop:
        images = conditional_preprocesses(images)
    else:
        images = v2.functional.resize(images, config['ViT']['image_size'])

    images = v2.functional.to_dtype(images, torch.float32, scale=True)

    if vae_prep:
        images = 2 * images - 1

    return images


class ImageDataset(Dataset):
    def __init__(self, maps, labels, image_path='./images/', transform=preprocessor, augmentation: bool = False, return_idx = False):
        self.image_path = image_path
        self.transform = transform
        self.return_idx = return_idx

        self.maps = pd.read_csv(maps)
        self.labels = pd.read_csv(labels)

        self.labels['realism'] = self.labels['fantasy'].map(lambda x: 1 - x)
        del self.labels['fantasy']

        self.labels['enmity'] = self.labels['romance'].map(lambda x: 1 - x)
        del self.labels['romance']

        self.augmentation = augmentation

        # force order and remove redactions
        self.labels = self.labels[config['Dataset']['filtered_cats']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.maps['cover_filename'][idx])
        image = Image.open(img_path).convert('RGB')

        label = self.labels.iloc[idx].values

        if self.transform:
            image = preprocessor(image, random_crop=self.augmentation)

        if self.return_idx:
            return image, torch.tensor(label, dtype=torch.float), idx

        return image, torch.tensor(label, dtype=torch.float)


training_dataset = ImageDataset('LNCovers/train/features.csv',
                                'LNCovers/train/labels.csv',
                                '../LNCovers/train/images/',
                                augmentation=True)

testing_dataset = ImageDataset('LNCovers/test/features.csv',
                               'LNCovers/test/labels.csv',
                               '../LNCovers/test/images/',
                               return_idx=True,
                               augmentation=False)
