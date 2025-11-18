# HUman DEScent

## Featured videos

- [Human Descent speed run — October 20, 2025](https://youtu.be/PIFJnmvqMIk)
- [Human Descent — How to play](https://youtu.be/tspa15Ei3KI)

**Live server:** The web version runs at [humandescent.net](https://humandescent.net) on an RTX 4090 GPU. Expect instability if many people connect simultaneously.

Human Descent opens the door for you to explore neural network optimization in high-dimensional weight space using the MNIST dataset. Manipulate everyday controllers (keyboard, xbox, DJ MIDI), each corresponding to a random direction in high dimensional weight space, to gain intuitive insights into the dynamics and complexities of neural network training.

[Demo video 2D/3D](https://youtu.be/VtF9dwoNQ_0)

### Older demo
[Demo video](https://youtu.be/mqAmaBP3-Q4)

## Game modes

- **3D landscape (default):** renders three coupled loss surfaces that update in real time while you explore six random directions.
- **1D slices (experimental):** renders six fast-updating loss curves, one per dimension. Launch with `?mode=1d` (e.g. `http://localhost:5173/?mode=1d`) or use the View toggle in the header to switch between modes.

## Batch-parameterized PyTorch layers

The visualizations in this project rely on a custom set of layers found in `hudes/model_first/model_first_nn.py`. These helpers wrap familiar PyTorch modules (linear, convolution, pooling, flatten, sequential) so that every forward pass accepts not just a minibatch of data, but also a *batch of parameter tensors*. Key pieces include:

- `get_param_module` and `MFSequential`, which convert a standard model definition into its batch-parameterized twin.
- `MFConv2d` and `MFLinear`, which reshape weights and biases on the fly to evaluate many parameter sets in parallel using grouped convolutions and batched einsum calls.
- Lightweight utilities like `MFMaxPool2d` and `Unsqueeze` that preserve the extra parameter dimension without copying data.

Because these modules keep the parameter batch dimension throughout the network, we can sweep across hundreds of points on the loss surface with a single forward pass. That efficiency is what makes the interactive loss-landscape exploration in the browser viable.

## Installation

```
git clone https://github.com/misko/human_descent.git # clone the repo
python3 -m venv hudes_env # create a virual enviornment
source hudes_env/bin/activate
cd human_descent
pip install . # install human descent and its dependencies in the virtual env
bash run.sh # download mnist, run the server, and then run the client!
```

## Details

The aim of this repository is to build an interactive tool to allow humans to directly
optimize high dimension problems such as neural network training. Inspired by [Measuring the Intrinsic Dimension of Objective Landscapes, (Chunyuan Li and
                  Heerad Farkhoor and
                  Rosanne Liu and
                  Jason Yosinski)](https://arxiv.org/abs/1804.08838)

A can use any of the ineractive inputs ([Xtouch mini midi mixer](https://www.amazon.com/gp/product/B013JLZCLS), [Xbox like controllers](https://www.amazon.com/gp/product/B091Y7HHS1), keyboard, etc..) to control a random n-dimensional subspace of parameter space at a time.

Iteratively selecting between new training batches and different random subspaces this allows you to optimize a very high dimensional weight space.

For example its possible to train a 26,000 parameter MNIST model using a 6 dimensional keyboard input in about 10 minutes.

![Example snapshot](images/01_demo.jpg)

## Testing

- Run all frontend checks with `cd web && npm test` (wrap in the repo's virtualenv so the backend can start).
- When iterating on the speed-run or highscores flows, capture deterministic Playwright logs with:

```
cd web
(npm run -s test:e2e:chromium -- tests/speedrun.spec.mjs) 2>&1 3>&1 | tee /tmp/play-speedrun.log >/dev/null
```

That command mirrors what we use in CI so any browser/server output is preserved under `/tmp/play-speedrun.log` for later debugging.
