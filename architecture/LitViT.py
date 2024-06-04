import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from sklearn.metrics import accuracy_score


@torch.no_grad()
def calc_accuracy(pred, target):
    """
    Calculate accuracy between predicted tensor and target tensor.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (1, N).
        target (torch.Tensor): Ground truth tensor of shape (1, N) in one-hot encoding.

    Returns:
        float: Accuracy value.
    """
    # Get the index of the highest value in the predicted tensor
    pred_idx = torch.argmax(pred, dim=1)

    # Get the index of the highest value (1) in the target tensor
    target_idx = torch.argmax(target, dim=1)

    # Compare the predicted and target indices
    correct = (pred_idx == target_idx).float()

    # Calculate the accuracy
    acc = correct.mean().item()

    return acc


class LitViT(L.LightningModule):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.vit(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)

        accuracy = calc_accuracy(y_hat, y)
        self.log('accuracy', accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.vit(x)

        accuracy = calc_accuracy(y_hat, y)
        self.log('val_accuracy', accuracy, on_epoch=True)
        return accuracy
