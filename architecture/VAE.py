from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
import yaml

config = yaml.safe_load(open('../constants/v2.yaml'))

model_path = hf_hub_download(config["VAE"]['hf_repo'],
                             config["VAE"]['hf_file_name'],
                             subfolder=config["VAE"]['hf_file_subdir'],
                             local_dir='models/vae')

model = AutoencoderKL.from_single_file(model_path)

def get_vae():
    return model