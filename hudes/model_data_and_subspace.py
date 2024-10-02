import math
from functools import cache
from multiprocessing import Queue

import torch
from torch import nn

MAX_DIMS = 2**16


class DatasetBatcher:
    def __init__(self, ds, batch_size: int, seed: int = 0):
        self.len = math.ceil(len(ds) / batch_size)
        self.batch_size = batch_size
        self.ds = ds
        g = torch.Generator()
        g.manual_seed(seed)
        self.idxs = torch.randperm(len(self.ds), generator=g)

    # TODO optionally cache this!
    @cache
    def __getitem__(self, idx: int):
        print("BATCHER", idx)
        idx = idx % self.len
        start_idx = idx * self.batch_size
        end_idx = min(len(self.ds), start_idx + self.batch_size)
        x, y = torch.cat(
            [self.ds[idx][0] for idx in self.idxs[start_idx:end_idx]], dim=0
        ), torch.tensor([self.ds[idx][1] for idx in self.idxs[start_idx:end_idx]])
        return {"data": x, "label": y}


# https://stackoverflow.com/questions/74865438/how-to-get-a-flattened-view-of-pytorch-model-parameters
def fuse_parameters(model: nn.Module, device):
    """Move model parameters to a contiguous tensor, and return that tensor."""
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n, device=device)
    i = 0
    for p in model.parameters():
        params_slice = params[i : i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params


def indexed_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return pred[torch.arange(label.shape[0]), label]


def get_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor):
    # (Pdb) preds.shape
    # torch.Size([512, 10])
    # (Pdb) labels.shape
    # torch.Size([512])
    assert preds.ndim == 2 and labels.ndim == 1
    n = preds.shape[1]
    c_matrix = torch.vstack(
        [
            torch.zeros(10, device=preds.device).scatter_(
                0, preds.argmax(dim=1)[labels == idx], 1, reduce="add"
            )
            for idx in torch.arange(n)
        ]
    )
    return torch.nn.functional.normalize(c_matrix, p=1, dim=1)


class ModelDataAndSubspace:

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        train_data_batcher: DatasetBatcher,
        val_data_batcher: DatasetBatcher,
        seed: int = 0,
        minimize: bool = False,
        device="cpu",
    ):
        self.device = device
        self.model = model  # .to(self.device)
        self.num_params = sum([x.numel() for x in model.parameters()])
        print(f"MODEL WITH : {self.num_params} parameters")
        self.train_data_batcher = train_data_batcher
        self.val_data_batcher = val_data_batcher
        self.input_q = Queue()
        self.minimize = minimize

        # fuse all params
        # self.model_params = fuse_parameters(model)
        # copy original weights
        # self.saved_weights = self.model_params.detach().clone()
        self.fused = False  # Cant fuse before forking

        g = torch.Generator()
        g.manual_seed(seed)
        self.seeds_for_dims = torch.randint(
            size=(MAX_DIMS,), high=2**32, generator=g
        )  # should be good enough
        self.loss_fn = loss_fn
        self.return_n_preds = 5

    def move_to_device(self):
        self.model = self.model.to(self.device)

    def fuse(self):
        self.model_params = fuse_parameters(self.model, self.device)
        self.saved_weights = self.model_params.detach().clone()
        self.fused = True

    # TODO could optimize with one large chunk of shared memory? and slice it?
    @cache
    def blank_weight_vec(self):
        wv = torch.zeros(*self.model_params.shape, device=self.device)
        # wv.share_memory_()
        return wv

    # todo cache this?
    @cache
    @torch.no_grad
    def get_dim_vec(self, dim: int):
        assert self.fused
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seeds_for_dims[dim % MAX_DIMS].item())
        return torch.nn.functional.normalize(
            torch.rand(1, *self.model_params.shape, generator=g, device=self.device)
            - 0.5,
            p=2,
            dim=1,
        )

    # dims is a dictionary {dim:step_size}
    @torch.no_grad
    def delta_from_dims(self, dims: dict[int, float]):
        if len(dims) > 0:
            return torch.cat(
                [
                    self.get_dim_vec(d) * v for d, v in dims.items()
                ]  # , device=self.device
            ).sum(axis=0)
        else:
            return self.blank_weight_vec()

    @torch.no_grad
    def set_parameters(self, weights: torch.Tensor):
        assert self.fused
        # self.model_params.copy_(weights) # segfaults?
        self.model_params *= 0
        self.model_params += weights

    # todo cache this?
    @cache
    def get_batch(self, idx: int):
        r = {}
        for name, batcher in (
            ("train", self.train_data_batcher),
            ("val", self.val_data_batcher),
        ):
            batch = batcher[idx]
            r[name] = (batch["data"].to(self.device), batch["label"].to(self.device))
        return r

    @torch.no_grad
    def val_model_inference_with_delta_weights(self, delta_weights: torch.Tensor):
        assert self.fused
        self.set_parameters(self.saved_weights + delta_weights)
        full_val_loss = 0
        n = 0
        for batch_idx in range(self.val_data_batcher.len):
            batch = self.get_batch(batch_idx)
            model_output = self.model(batch["val"][0])
            full_val_loss += self.loss_fn(model_output, batch["val"][1]).sum().item()
            n += batch["val"][1].shape[0]
        if not self.minimize:
            full_val_loss = -full_val_loss
        return {"val_loss": full_val_loss / n}

    @torch.no_grad
    def train_model_inference_with_delta_weights(
        self, delta_weights: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        assert self.fused
        batch = self.get_batch(batch_idx)
        self.set_parameters(self.saved_weights + delta_weights)
        model_output = self.model(batch["train"][0])
        train_loss = self.loss_fn(model_output, batch["train"][1]).mean().item()
        train_pred = self.model.probs(model_output)
        if not self.minimize:
            train_loss = -train_loss

        confusion_matrix = get_confusion_matrix(train_pred, batch["train"][1])
        return {
            "train_loss": train_loss,
            "train_preds": train_pred[: self.return_n_preds].cpu(),
            "confusion_matrix": confusion_matrix.cpu(),
        }
