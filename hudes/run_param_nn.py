import torch

from hudes.mnist import MNISTFFNN, MNISTParamNet, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import ParamModelDataAndSubspace, indexed_loss

if __name__ == "__main__":
    device = "mps"
    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN(),
        loss_fn=indexed_loss,
        device=device,
        constructor=ParamModelDataAndSubspace,
    )
    mad.move_to_device()
    mad.fuse()
    mad.init_param_model()
    mp = mad.dim_idxs_and_ranges_to_models_parms(
        [0, 10],
        arange=torch.linspace(-1, 1, 50, device=device),
        brange=torch.linspace(-1, 1, 100, device=device),
    )

    mp_reshaped = mp.reshape(-1, 26506)
    batch = torch.rand(1, 512, 28, 28, device=device)

    for i in range(50):
        mad.param_model.forward(mp_reshaped, batch)
