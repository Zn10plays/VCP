import torch
import numpy as np
import yaml
from torchvision import transforms as t

config = yaml.safe_load(open('constants/v1.yaml'))

classes = np.array(config['Dataset']['classes'])


# given a torch array, return the index of the top 5 logits
def pick_top(x: torch.Tensor, count=1, sorted: bool = True):
    return x.topk(k=count, sorted=sorted)


# return the parsed input
def prettify(x):
    return classes[x.indices.to('cpu')].squeeze().tolist(), x.values.to('cpu').squeeze().tolist()

def to_classes(x: torch.Tensor):
    count = int(x.sum())
    return prettify(pick_top(x, count=count))

to_pil = t.ToPILImage()