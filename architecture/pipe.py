import torch
from ViT import SimpleViT
from VAE import get_vae
import warnings
import PIL

import yaml

config = yaml.safe_load(open('constants/v2.yaml'))


def get_model(ver: int = 2):
    if ver == 1:
        model = None
        # implement v1 and return
        warnings.warn('not implemented cus it sucked')
    elif ver == 2:
        model = SimpleViT(
            image_size=(int(config['Model']['image_size'][0] / 8), int(config['Model']['image_size'][1] / 8)),
            patch_size=config['Model']['patch_size'],
            num_classes=len(config['Model']['classes']) - len(config['Model']['redactions']),
            dim=config['Model']['dim_size'],
            depth=config['Model']['depth'],
            heads=config['Model']['heads'],
            mlp_dim=config['Model']['mlp_dim'],
        )

    return model


model = get_model()
vae = get_vae()

@torch.amp.autocast('cuda')
def pipe(images, return_logits=True):
    with torch.no_grad():
        prior = vae.encode(images)
        samples = prior.latent_dist.sample()

    logits = model(samples)
    return logits if return_logits else logits.softmax(-1)