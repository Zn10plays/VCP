import torch
from PIL import Image

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as t

preprocessor = t.Compose([
    t.Resize((128 * 3, 128 * 2)),
    t.PILToTensor(),
    t.ConvertImageDtype(torch.float)
])


class ImageDataset(Dataset):
    def __init__(self, maps, labels, image_path='./images/', transform=preprocessor):
        self.maps = pd.read_csv(maps)
        self.labels = pd.read_csv(labels)
        del self.labels['Unnamed: 0']
        self.image_path = image_path

        self.transform = transform

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
