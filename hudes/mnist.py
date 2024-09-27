import math
import os
from functools import cache
from multiprocessing import Queue

import torch
from torch import nn
from torchvision import datasets, transforms

from model_data_and_subspace import DatasetBatcher, ModelDataAndSubspace, indexed_loss


class MNISTFFNN(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        mnist_width_height = 28
        mnist_classes = 10
        hidden = 32
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mnist_width_height * mnist_width_height, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, mnist_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)

    def probs(self, x: torch.Tensor):
        return x.exp()


class MNISTCNN(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x[:, None])
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.sm(self.out(x))
        return output  # return x for visualization

    def probs(self, x: torch.Tensor):
        return x.exp()


# precompute these for K seeds
def subspace_basis(model, seed: int, dim: int):
    torch.manual_seed(seed)
    return [torch.rand(dim, *x.shape) - 0.5 for x in model.parameters()]


@torch.no_grad
def set_parameters(model, weights: torch.Tensor):
    idx = 0
    for param in model.parameters():
        # assert param.shape == weights[idx].shape
        # param.copy_(weights[idx]) # this segfaults (?)
        param *= 0  # this is worse performance :'(
        param += weights[idx]
        idx += 1


def mnist_model_data_and_subpace(
    model: nn.Module,
    seed: int = 0,
    store: str = "./",
    train_batch_size: int = 512,
    val_batch_size: int = 1024,
    loss_fn=indexed_loss,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
        ]
    )

    train_data_batcher = DatasetBatcher(
        datasets.MNIST(
            os.path.join(store, "mnist_data"),
            download=True,
            train=True,
            transform=transform,
        ),
        train_batch_size,
        seed=seed,
    )
    val_data_batcher = DatasetBatcher(
        datasets.MNIST(
            os.path.join(store, "mnist_data"),
            download=True,
            train=False,
            transform=transform,
        ),
        val_batch_size,
        seed=seed,
    )
    return ModelDataAndSubspace(
        model=model,
        train_data_batcher=train_data_batcher,
        val_data_batcher=val_data_batcher,
        loss_fn=indexed_loss,
        minimize=False,
    )
