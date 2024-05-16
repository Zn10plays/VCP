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
def prettify_classes(x):
    return classes[x.indices.to('cpu')].squeeze().tolist(), x.values.to('cpu').squeeze().tolist()


def to_classes(x: torch.Tensor):
    count = int(x.sum())
    return prettify_classes(pick_top(x, count=count))


to_pil = t.ToPILImage()


@torch.no_grad()
def calculate_precession(y_hat: torch.Tensor, y: torch.Tensor, pooling: bool = False):
    y = y.cpu()
    y_hat = y_hat.cpu()

    classes_hat = y_hat.apply_(lambda x: 1 if x > .7 else 0)
    y_complement = y.apply_(lambda x: 0 if x == 1 else 0)
    
    acc = ((cls * y).sum(dim = -1) - (cls * y_composite).sum(dim = -1)) / y.sum(dim=-1)
    
    return torch.mean(acc) if pooling else acc
