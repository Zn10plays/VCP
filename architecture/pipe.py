import torch
from .ViT import SimpleViT
from .VAE import get_vae
import warnings
import PIL

import yaml

config = yaml.full_load(open('constants/v2.yaml'))


def get_model(ver: int = 2, device: str = 'cuda'):
    if ver == 1:
        vit = None
        # implement v1 and return
        warnings.warn('not implemented cus it sucked')
        return vit
    elif ver == 2:
        vit = SimpleViT(
            image_size=(int(config['ViT']['image_size'][0] / 8), int(config['ViT']['image_size'][1] / 8)),
            patch_size=config['ViT']['patch_size'],
            num_classes=len(config['Dataset']['classes']) - len(config['Dataset']['redactions']),
            dim=config['ViT']['dim_size'],
            depth=config['ViT']['depth'],
            heads=config['ViT']['heads'],
            mlp_dim=config['ViT']['mlp_dim'],
        )
        return vit.to(device)


model = get_model()
vae = get_vae()


@torch.amp.autocast('cuda')
def pipe(images, return_logits=True):
    with torch.no_grad():
        prior = vae.encode(images)
        samples = prior.latent_dist.sample()

    logits = model(samples)
    return logits if return_logits else logits.softmax(-1)
