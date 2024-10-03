import argparse
import time

import torch

from hudes.mnist import MNISTFFNN, mnist_model_data_and_subpace
from hudes.model_data_and_subspace import indexed_loss

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
    )
    mad.move_to_device()
    mad.fuse()
    mad.init_param_model()

    mp = mad.dim_idxs_and_ranges_to_models_parms(
        base_weights=mad.saved_weights,
        dims=[0, 10],
        arange=torch.linspace(-1, 1, args.grid, device=args.device),
        brange=torch.linspace(-1, 1, args.grid, device=args.device),
    )

    mp_reshaped = mp.reshape(-1, 26506)
    batch = torch.rand(1, 512, 28, 28, device=args.device, dtype=dtype)
    label = torch.randint(low=0, high=10, size=(512,), device=args.device)

    start_time = time.time()
    x = 0
    for i in range(args.n):
        predictions = mad.param_model.forward(mp_reshaped, batch)[1].reshape(
            *mp.shape[:2], 512, -1
        )
        loss = torch.gather(
            predictions,
            3,
            label.reshape(1, 1, -1, 1).expand(*mp.shape[:2], 512, 1),
        ).mean(axis=[2, 3])
        x += loss.cpu().mean()
        print(predictions.shape, label.shape)
    print(f"{(time.time()-start_time)/args.n:0.4e}s per iteration")
