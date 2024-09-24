from functools import cache
from multiprocessing import Queue

import torch

MAX_DIMS = 2**16


# https://stackoverflow.com/questions/74865438/how-to-get-a-flattened-view-of-pytorch-model-parameters
def fuse_parameters(model):
    """Move model parameters to a contiguous tensor, and return that tensor."""
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n)
    i = 0
    for p in model.parameters():
        params_slice = params[i : i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params


def indexed_loss(pred, label):
    return pred[torch.arange(label.shape[0]), label]


def get_confusion_matrix(preds, labels):
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

    def __init__(self, model, loss_fn, train_data_batcher, val_data_batcher, seed=0):
        self.model = model
        self.num_params = sum([x.numel() for x in model.parameters()])
        print(f"MODEL WITH : {self.num_params} parameters")
        self.train_data_batcher = train_data_batcher
        self.val_data_batcher = val_data_batcher
        self.input_q = Queue()

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

    def fuse(self):
        self.model_params = fuse_parameters(self.model)
        self.saved_weights = self.model_params.detach().clone()
        self.fused = True

    # TODO could optimize with one large chunk of shared memory? and slice it?
    def blank_weight_vec(self):
        wv = torch.zeros(*self.model_params.shape)
        wv.share_memory_()
        return wv

    # todo cache this?
    @cache
    @torch.no_grad
    def get_dim_vec(self, dim):
        assert self.fused
        g = torch.Generator()
        g.manual_seed(self.seeds_for_dims[dim % MAX_DIMS].item())
        return torch.rand(1, *self.model_params.shape, generator=g) - 0.5

    # dims is a dictionary {dim:step_size}
    @torch.no_grad
    def delta_from_dims(self, dims):
        return torch.cat([self.get_dim_vec(d) * v for d, v in dims.items()]).sum(axis=0)

    @torch.no_grad
    def set_parameters(self, weights):
        assert self.fused
        # self.model_params.copy_(weights) # segfaults?
        self.model_params *= 0
        self.model_params += weights

    # todo cache this?
    @cache
    def get_batch(self, idx):
        return {
            "train": self.train_data_batcher[idx],
            "val": self.val_data_batcher[idx],
        }

    @torch.no_grad
    def val_model_inference_with_delta_weights(self, delta_weights):
        assert self.fused
        self.set_parameters(self.saved_weights + delta_weights)
        full_val_loss = 0
        n = 0
        for batch_idx in range(self.val_data_batcher.len):
            batch = self.get_batch(batch_idx)
            model_output = self.model(batch["val"][0])
            full_val_loss = self.loss_fn(model_output, batch["val"][1]).sum().item()
            n += batch["val"][1].shape[0]
        if not self.model.minimize:
            full_val_loss = -full_val_loss
        return {"val_loss": full_val_loss / n}

    @torch.no_grad
    def train_model_inference_with_delta_weights(self, delta_weights, batch_idx):
        assert self.fused
        batch = self.get_batch(batch_idx)
        self.set_parameters(self.saved_weights + delta_weights)
        model_output = self.model(batch["train"][0])
        train_loss = self.loss_fn(model_output, batch["train"][1]).mean().item()
        train_pred = self.model.probs(model_output)
        if not self.model.minimize:
            train_loss = -train_loss

        confusion_matrix = get_confusion_matrix(train_pred, batch["train"][1])
        return {
            "train_loss": train_loss,
            "train_preds": train_pred[: self.return_n_preds],
            "confusion_matrix": confusion_matrix,
        }
