import torch
from .ViT import SimpleViT
from dataloader.original import preprocessor
from .LitViT import LitViT
import yaml


def get_model(ver='original'):
    if ver == 'original':
        config = yaml.full_load(open('constants/original.yaml'))

        model = SimpleViT(
            image_size=(config['ViT']['image_size'][0], config['ViT']['image_size'][1]),
            patch_size=config['ViT']['patch_size'],
            num_classes=len(config['Dataset']['genre']),
            dim=config['ViT']['dim_size'],
            depth=config['ViT']['depth'],
            heads=config['ViT']['heads'],
            mlp_dim=config['ViT']['mlp_dim'],
        )

        return LitViT(model)
    else:
        config = yaml.full_load(open('constants/v2.yaml'))
        model = SimpleViT(
            image_size=(config['ViT']['image_size'][0], config['ViT']['image_size'][1]),
            patch_size=config['ViT']['patch_size'],
            num_classes=len(config['Dataset']['filtered_cats']),
            dim=config['ViT']['dim_size'],
            depth=config['ViT']['depth'],
            heads=config['ViT']['heads'],
            mlp_dim=config['ViT']['mlp_dim'],
        )

        return LitViT(model)



@torch.amp.autocast('cuda')
def pipe(images, model, return_logits=False):
    images = preprocessor(images, random_crop=False)

    logits = model(images)
    return logits if return_logits else logits.sigmoid()
