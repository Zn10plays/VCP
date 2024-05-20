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
def calculate_precession(y_hat: torch.Tensor, y: torch.Tensor, pooling = False, cutoff=.5):
    y = y.cpu().clone()
    y_hat = y_hat.cpu().clone()

    classes_hat = y_hat.clone().apply_(lambda x: 1 if x > .35 else 0)
    y_complement = y.clone().apply_(lambda x: 0 if x == 1 else 1)

    acc = ((classes_hat * y).sum(dim = -1) - (y_complement * classes_hat).sum(dim=-1))

    return torch.mean(acc) if pooling else acc

def _prob_squeeze(x: torch.Tensor):
    return -0.00034 * torch.pow(x, 2) + .5 if x < .5 else .5 * torch.pow(x, 2) + .5

