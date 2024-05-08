import torch
from vit_pytorch import ViT
from config import num_classes


vision_model = ViT(
    image_size=128 * 3,
    patch_size=32,
    num_classes=num_classes,
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

