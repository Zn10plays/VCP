import torch
from .ViT import SimpleViT
from dataloader import preprocessor
import yaml

config = yaml.full_load(open('constants/v2.yaml'))


def get_model(device: str = 'cuda'):
    return SimpleViT(
        image_size=(config['ViT']['image_size'][0], config['ViT']['image_size']),
        patch_size=config['ViT']['patch_size'],
        num_classes=len(config['Dataset']['filtered_cats']),
        dim=config['ViT']['dim_size'],
        depth=config['ViT']['depth'],
        heads=config['ViT']['heads'],
        mlp_dim=config['ViT']['mlp_dim'],
    ).to(device)

@torch.amp.autocast('cuda')
def pipe(images, model, return_logits=False):

    images = preprocessor(images, random_crop=False)

    logits = model(images)
    return logits if return_logits else logits.sigmoid()
