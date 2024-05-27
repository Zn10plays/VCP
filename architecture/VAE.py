import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
import yaml

config = yaml.full_load(open('constants/v2.yaml'))['VAE']

model_url = config['hf_model_url']

model = AutoencoderKL.from_single_file(model_url, cache_dir='./models/vae')


def get_vae(device: torch.device = 'cuda') -> AutoencoderKL:
    return model.to(device)
