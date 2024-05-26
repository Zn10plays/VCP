import torch
from ViT import SimpleViT
from VAE import get_vae
import PIL

import yaml

config = yaml.safe_load(open('constants/v2.yaml'))
vae = get_vae()

def pipe(images):
    prior = vae.encode(images)

