import copy
import logging
import math
from functools import cache

import torch
from torch import nn

from hudes.model_first import model_first_nn

MAX_DIMS = 2**16


class DatasetBatcher:
    def __init__(self, ds, seed: int = 0):
        self.ds = ds
        self.seed = seed
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.idxs = torch.randperm(len(self.ds), generator=g)

    @cache
    def get_len(self, batch_size: int):
        return math.ceil(len(self.ds) / batch_size)

    # TODO optionally cache this!
    @cache
    def get_batch(self, batch_size, batch_idx):
        logging.debug(f"get_batch size: {self} {batch_size} idx: {batch_idx}")
        batch_idx = batch_idx % self.get_len(batch_size=batch_size)
        start_idx = batch_idx * batch_size
        end_idx = min(len(self.ds), start_idx + batch_size)
        x, y = torch.cat(
            [self.ds[idx][0] for idx in self.idxs[start_idx:end_idx]], dim=0
        ), torch.tensor([self.ds[idx][1] for idx in self.idxs[start_idx:end_idx]])
        return {"data": x, "label": y}


# https://stackoverflow.com/questions/74865438/how-to-get-a-flattened-view-of-pytorch-model-parameters
def fuse_parameters(model: nn.Module, device, dtype):
    """Move model parameters to a contiguous tensor, and return that tensor."""
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n, device=device, dtype=dtype)
    i = 0
    for p in model.parameters():
        params_slice = params[i : i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params


@torch.jit.script
def jit_indexed_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return pred[torch.arange(label.shape[0]), label]


def indexed_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return jit_indexed_loss(pred, label)


def get_param_module(module):
    if isinstance(module, torch.nn.Flatten):
        return torch.nn.Flatten(start_dim=module.start_dim + 1)
    if isinstance(module, torch.nn.LogSoftmax):
        return torch.nn.LogSoftmax(dim=module.dim + 1)
    if not hasattr(module, "parameters") or len(list(module.parameters())) == 0:
        return module
    if isinstance(module, torch.nn.Linear):
        out_channels, in_channels = module.weight.shape
        return model_first_nn.Linear(
            input_channels=in_channels, output_channels=out_channels
        )
    raise ValueError(type(module))


def param_nn_from_sequential(model):
    return model_first_nn.Sequential([get_param_module(m) for m in model])


@torch.jit.script
def get_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor):
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
    return torch.nn.functional.normalize(c_matrix, p=1.0, dim=1)


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
        val_batch_size: int = 1024,
    ):
        self.val_batch_size = val_batch_size
        self.device = device
        self._model = model  # .to(self.device)
        self.num_params = sum([p.numel() for p in self._model.parameters()])
        self.batchers = {"train": train_data_batcher, "val": val_data_batcher}
        self.minimize = minimize

        self.fused = False  # Cant fuse before forking

        g = torch.Generator()
        g.manual_seed(seed)
        self.seeds_for_dims = torch.randint(
            size=(MAX_DIMS,), high=2**32, generator=g
        )  # should be good enough
        self.loss_fn = loss_fn
        self.return_n_preds = 5
        self.models = {
            torch.float32: copy.deepcopy(self._model).to(torch.float32),
            torch.float16: copy.deepcopy(self._model).to(torch.float16),
        }

    def move_to_device(self):
        self.models = {k: v.to(self.device) for k, v in self.models.items()}

    def fuse(self):
        self.model_params = {
            torch.float32: fuse_parameters(
                self.models[torch.float32], self.device, torch.float32
            ),
            torch.float16: fuse_parameters(
                self.models[torch.float16], self.device, torch.float16
            ),
        }
        self.saved_weights = {
            torch.float32: self.model_params[torch.float32].detach().clone(),
            torch.float16: self.model_params[torch.float16].detach().clone(),
        }
        self.fused = True

    # todo cache this?
    @cache
    @torch.no_grad
    def get_dim_vec(self, dim: int, dtype):
        assert self.fused
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seeds_for_dims[dim % MAX_DIMS].item())
        return torch.nn.functional.normalize(
            torch.rand(
                1,
                *self.model_params[dtype].shape,
                generator=g,
                device=self.device,
                dtype=torch.float32,
            )
            - 0.5,
            p=2,
            dim=1,
        ).to(dtype)

    @torch.no_grad
    def delta_from_dims(self, dims: dict[int, float], dtype: torch.dtype):
        if len(dims) > 0:
            return torch.cat(
                [self.get_dim_vec(d, dtype=dtype) * v for d, v in dims.items()]
            ).sum(axis=0)
        else:
            return self.blank_weight_vec()

    # todo cache this?
    @cache
    def get_batch(self, batch_size: int, batch_idx: int, dtype, train_or_val: str):
        assert train_or_val in self.batchers
        batch = self.batchers[train_or_val].get_batch(batch_size, batch_idx)
        return (
            batch["data"].to(device=self.device, dtype=dtype),
            batch["label"].to(device=self.device),
        )

    # TODO could optimize with one large chunk of shared memory? and slice it?
    @cache
    def blank_weight_vec(self):
        return torch.zeros(*self.model_params[torch.float32].shape, device=self.device)

    @torch.no_grad
    def set_parameters(self, weights: torch.Tensor, dtype):
        assert self.fused
        self.model_params[dtype].data.copy_(weights)

    @torch.no_grad
    def val_model_inference_with_delta_weights(self, weights: torch.Tensor, dtype):
        assert self.fused
        self.set_parameters(weights, dtype)
        full_val_loss = 0
        n = 0
        for batch_idx in range(self.batchers["val"].get_len(self.val_batch_size)):
            batch = self.get_batch(
                batch_size=self.val_batch_size,
                batch_idx=batch_idx,
                dtype=dtype,
                train_or_val="val",
            )
            model_output = self.models[dtype](batch[0])
            full_val_loss += self.loss_fn(model_output, batch[1]).sum().item()
            n += batch[1].shape[0]
        if not self.minimize:
            full_val_loss = -full_val_loss
        return {"val_loss": full_val_loss / n}

    @torch.no_grad
    def train_model_inference_with_delta_weights(
        self, weights: torch.Tensor, batch_size: int, batch_idx: int, dtype
    ) -> dict[str, torch.Tensor]:
        assert self.fused
        batch = self.get_batch(batch_size, batch_idx, dtype=dtype, train_or_val="train")
        logging.debug(f"GOT batch {batch_size} {batch_idx} {batch[0].shape}")
        self.set_parameters(weights, dtype)
        model_output = self.models[dtype](batch[0])
        train_loss = self.loss_fn(model_output, batch[1]).mean().item()
        train_pred = self.models[dtype].probs(model_output)
        if not self.minimize:
            train_loss = -train_loss

        confusion_matrix = get_confusion_matrix(train_pred, batch[1])
        logging.info(
            f"train loss: {train_loss} MO:{model_output.mean()}/{model_output.shape} weights {weights.mean().item()} {dtype}"
        )
        logging.debug(f"train loss: total preds {train_pred.shape}")
        # breakpoint()
        return {
            "train_loss": train_loss,
            "train_preds": train_pred[: self.return_n_preds].cpu(),
            "confusion_matrix": confusion_matrix.cpu(),
        }

    def sgd_step(self, model_weights, n_steps, dtype, batch_size, batch_idx):
        # set the model parameters
        self.set_parameters(model_weights, dtype)
        if model_weights.device.type == "mps" and dtype != torch.float32:
            dtype = torch.float32
        # figure out if we need to make a new optimizer
        model = self.models[dtype]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        for _ in range(n_steps):
            batch = self.get_batch(
                batch_size, batch_idx, dtype=dtype, train_or_val="train"
            )
            model_output = model(batch[0])
            loss = -self.loss_fn(model_output, batch[1]).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model_weights.data.copy_(self.model_params[dtype])
        return (
            self.train_model_inference_with_delta_weights(
                model_weights, batch_size, batch_idx, dtype
            ),
            model_weights,
        )

    def init_param_model(self):
        self.param_models = {
            torch.float32: param_nn_from_sequential(self.models[torch.float32].net),
            torch.float16: param_nn_from_sequential(self.models[torch.float16].net),
        }

    # return model parameters for given ranges
    def dim_idxs_and_ranges_to_models_parms(
        self,
        base_weights,
        dims: torch.Tensor,
        arange: torch.Tensor,
        brange: torch.Tensor,
        dtype: torch.dtype,
    ):
        assert len(dims) == 2
        vs = torch.vstack([self.get_dim_vec(dim, dtype=dtype) for dim in dims])

        agrid, bgrid = torch.meshgrid(torch.tensor(arange), torch.tensor(brange))
        agrid = agrid.unsqueeze(2)
        bgrid = bgrid.unsqueeze(2)
        return torch.concatenate([agrid, bgrid], dim=2).to(
            dtype
        ) @ vs + base_weights.reshape(1, 1, -1)

    def get_loss_grid(
        self,
        base_weights,
        batch_idx,
        dims_offset,
        grids,
        grid_size,
        step_size,
        batch_size,
        dtype,
    ):
        assert grid_size % 2 == 1
        assert grid_size > 3
        assert grids > 0

        batch = self.get_batch(
            batch_size=batch_size,
            batch_idx=batch_idx,
            dtype=dtype,
            train_or_val="train",
        )
        data = batch[0].unsqueeze(0)
        batch_size = data.shape[1]
        label = batch[1]
        r = (torch.arange(grid_size, device=self.device) - grid_size // 2) * step_size

        grid_losses = []
        for grid_idx in range(grids):
            dims = [
                dims_offset + grid_idx * 2,
                dims_offset + grid_idx * 2 + 1,
            ]  # which dims are doing this for

            mp = self.dim_idxs_and_ranges_to_models_parms(
                base_weights, dims, arange=r, brange=r, dtype=dtype
            )

            mp_reshaped = mp.reshape(-1, self.num_params).contiguous()
            predictions = (
                self.param_models[dtype]
                .forward(mp_reshaped, data)[1]
                .reshape(*mp.shape[:2], batch_size, -1)
            )
            loss = torch.gather(
                predictions,
                3,
                label.reshape(1, 1, -1, 1).expand(*mp.shape[:2], batch_size, 1),
            ).mean(axis=[2, 3])
            loss = -loss
            grid_losses.append(loss.unsqueeze(0))
        loss = torch.concatenate(grid_losses, dim=0).cpu()
        logging.info(f"get_loss: return loss {loss[:, grid_size // 2, grid_size // 2]}")
        return loss
