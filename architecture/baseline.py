import torch
from vit_pytorch import ViT
import yaml

config = yaml.safe_load(open('constants/v1.yaml'))

vision_model = ViT(
    image_size=max(config['model']['image_size']),
    patch_size=32,
    num_classes=len(config['model']['classes']),
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

