import argparse
import time

import torch

from hudes.mnist import MNISTFFNN, MNISTParamNet, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import ParamModelDataAndSubspace, indexed_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hudes: Keyboardclient")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--n", type=int, default=50)

    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    mad = mnist_model_data_and_subpace(
        model=MNISTFFNN().to(dtype),
        loss_fn=indexed_loss,
        device=args.device,
        dtype=dtype,
        constructor=ParamModelDataAndSubspace,
    )
    mad.move_to_device()
    mad.fuse()
    mad.init_param_model()

    mp = mad.dim_idxs_and_ranges_to_models_parms(
        [0, 10],
        arange=torch.linspace(-1, 1, args.grid, device=args.device),
        brange=torch.linspace(-1, 1, args.grid, device=args.device),
    )

    mp_reshaped = mp.reshape(-1, 26506)
    batch = torch.rand(1, 512, 28, 28, device=args.device, dtype=dtype)

    start_time = time.time()
    for i in range(args.n):
        mad.param_model.forward(mp_reshaped, batch)
    print(f"{(time.time()-start_time)/args.n:0.4e}s per iteration")
