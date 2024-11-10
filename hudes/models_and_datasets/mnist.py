import os

import torch
from torch import nn
from torchvision import datasets, transforms

from hudes.model_data_and_subspace import (
    DatasetBatcher,
    ModelDataAndSubspace,
    indexed_loss,
)
from hudes.model_first.model_first_nn import (
    MFConv2d,
    MFFlip,
    MFLinear,
    MFMaxPool2d,
    MFSequential,
    Unsqueeze,
)


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


class MNISTCNN3(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output 10 classes
        self.lin1 = nn.Linear(8 * 7 * 7, 32)
        self.lin2 = nn.Linear(32, 10)
        self.sm = nn.LogSoftmax(dim=1)
        self.flatten = torch.nn.Flatten(1)
        self.unsqueeze = Unsqueeze(1)

        self.net = torch.nn.Sequential(
            self.unsqueeze,  # out (b,1,28,28)
            self.conv1,  # out (b,8,14,14)
            self.flatten,  # out (b,8,14,14)
            self.lin1,
            nn.ReLU(),
            self.lin2,
            self.sm,
        )

    def forward(self, x):
        return self.net(x)

    def probs(self, x: torch.Tensor):
        return x.exp()


def MNISTCNNFlipped():
    return MFSequential(
        [
            Unsqueeze(2),
            # Flip here,
            MFFlip(),
            MFConv2d(
                input_channels=1,
                output_channels=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
                flipped=True,
            ),
            nn.ReLU(),
            MFMaxPool2d(kernel_size=2, stride=2, padding=0),
            MFConv2d(
                input_channels=16,
                output_channels=32,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
                flipped=True,
            ),
            nn.ReLU(),
            MFMaxPool2d(kernel_size=2, stride=2, padding=0),
            MFFlip(),
            # Need to unflip here
            torch.nn.Flatten(2),
            MFLinear(32 * 7 * 7, 10),
            nn.LogSoftmax(dim=2),
        ]
    )


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
        self.flatten = torch.nn.Flatten(1)
        self.unsqueeze = Unsqueeze(1)

        self.net = torch.nn.Sequential(
            self.unsqueeze, self.conv1, self.conv2, self.flatten, self.out, self.sm
        )

    def forward(self, x):
        return self.net(x)

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
    model: nn.Module, seed: int = 0, store: str = "./", device="cpu", param_models=None
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
        seed=seed,
    )
    val_data_batcher = DatasetBatcher(
        datasets.MNIST(
            os.path.join(store, "mnist_data"),
            download=True,
            train=False,
            transform=transform,
        ),
        seed=seed,
    )
    return ModelDataAndSubspace(
        model=model,
        train_data_batcher=train_data_batcher,
        val_data_batcher=val_data_batcher,
        loss_fn=indexed_loss,
        minimize=False,
        device=device,
        param_models=param_models,
    )
