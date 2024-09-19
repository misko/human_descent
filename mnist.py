import math
from torchvision import datasets
from torchvision import transforms
from torch import nn
import torch
import os

from multiprocessing import Queue

from functools import cache


class DatasetBatcher:
    def __init__(self, ds, batch_size):
        self.len = math.ceil(len(ds) / batch_size)
        self.batch_size = batch_size
        self.ds = ds
        self.idxs = torch.randperm(len(self.ds))

    # TODO optionally cache this!
    @cache
    def __getitem__(self, idx):
        if idx > self.len:
            raise ValueError(f"Batch {idx} is out of bounds [0,{self.len}]")
        start_idx = idx * self.batch_size
        end_idx = min(len(self.ds), start_idx + self.batch_size)
        x, y = torch.cat(
            [self.ds[idx][0] for idx in self.idxs[start_idx:end_idx]], dim=0
        ), torch.tensor([self.ds[idx][1] for idx in self.idxs[start_idx:end_idx]])
        return x, y


class MNISTFFNN(nn.Module):
    def __init__(self):
        super().__init__()
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


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
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


# precompute these for K seeds
def subspace_basis(model, seed, dim):
    torch.manual_seed(seed)
    return [torch.rand(dim, *x.shape) - 0.5 for x in model.parameters()]


@torch.no_grad
def set_parameters(model, weights):
    idx = 0
    for param in model.parameters():
        # assert param.shape == weights[idx].shape
        # param.copy_(weights[idx]) # this segfaults (?)
        param *= 0  # this is worse performance :'(
        param += weights[idx]
        idx += 1


class ModelAndData:
    def __init__(self, model, train_batch_size, val_batch_size, store="./"):
        self.model = model

        torch.manual_seed(0)

        self.num_params = sum([x.numel() for x in model.parameters()])
        print(f"MODEL WITH : {self.num_params} parameters")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        )

        self.train_data_batcher = DatasetBatcher(
            datasets.MNIST(
                os.path.join(store, "mnist_data"),
                download=True,
                train=True,
                transform=transform,
            ),
            train_batch_size,
        )
        self.val_data_batcher = DatasetBatcher(
            datasets.MNIST(
                os.path.join(store, "mnist_data"),
                download=True,
                train=False,
                transform=transform,
            ),
            train_batch_size,
        )
        self.input_q = Queue()

    def get_batch(self, idx):
        return {
            "train": self.train_data_batcher[idx],
            "val": self.val_data_batcher[idx],
        }
