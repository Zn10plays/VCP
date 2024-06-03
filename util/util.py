import torch
import numpy as np
import yaml


class LabelParser:
    def __init__(self, ver='original'):
        self.ver = ver

        if ver == 'original':
            self.config = yaml.full_load(open('constants/original.yaml'))
            self.classes = np.array(self.config['Dataset']['genre'])

        if ver == 'v2':
            self.config = yaml.full_load(open('constants/v2.yaml'))
            self.classes = np.array(self.config['Dataset']['filtered_cats'])

    def pick_top(self, x: torch.Tensor, count=1, sorted: bool = True, prettify: bool = False):
        return x.topk(k=count, sorted=sorted) if not prettify else self.prettify_classes(x.topk(k=count, sorted=sorted))

    # return the parsed input
    def prettify_classes(self, x):
        return self.classes[x.indices.to('cpu')].squeeze().tolist(), x.values.to('cpu').squeeze().tolist()

    def to_classes(self, x: torch.Tensor):
        count = int(x.sum())
        return self.prettify_classes(self.pick_top(x, count=count))

@torch.no_grad()
def calculate_precession(y_hat: torch.Tensor, y: torch.Tensor, pooling=False, cutoff=.35):
    y = y.cpu().clone()
    y_hat = y_hat.cpu().clone()

    classes_hat = y_hat.clone().apply_(lambda x: 1 if x > cutoff else 0)
    y_complement = y.clone().apply_(lambda x: 0 if x == 1 else 1)

    acc = ((classes_hat * y).sum(dim=-1) - (y_complement * classes_hat).sum(dim=-1)) / torch.sum(y, dim=-1)

    return torch.mean(acc) if pooling else acc


def _prob_squeeze(x: torch.Tensor):
    return -0.00034 * torch.pow(x, 2) + .5 if x < .5 else .5 * torch.pow(x, 2) + .5
