import torch

from hudes.model_data_and_subspace import param_nn_from_sequential
from hudes.model_first.model_first_nn import MFLinear, MFSequential
from hudes.models_and_datasets.mnist import MNISTCNN


def test_linear_and_relu():

    in_channels = 3
    out_channels = 5
    batch_size = 7
    n_models = 3
    model_params = torch.rand([n_models, in_channels * out_channels + out_channels])

    batch = torch.rand(1, batch_size, in_channels).expand(
        n_models, batch_size, in_channels
    )

    _mp, out = MFLinear(in_channels, out_channels).forward(model_params, batch)
    out = torch.nn.functional.relu(out)

    outputs = []
    for model_idx in range(n_models):
        _weight = model_params[model_idx, : in_channels * out_channels].reshape(
            out_channels, in_channels
        )
        _bias = model_params[model_idx, in_channels * out_channels :]
        layer = torch.nn.Linear(in_channels, out_channels)
        layer.weight.data.copy_(_weight)
        layer.bias.data.copy_(_bias)
        outputs.append(torch.nn.functional.relu(layer(batch[0])).unsqueeze(0))

    assert torch.vstack(outputs).isclose(out).all()


def test_mnist():
    mnist_width_height = 28
    mnist_classes = 10
    hidden = 32
    mnist_net = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(mnist_width_height * mnist_width_height, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, mnist_classes),
        torch.nn.LogSoftmax(dim=1),
    )

    mnist_param_net = MFSequential(
        [
            torch.nn.Flatten(start_dim=2),
            MFLinear(mnist_width_height * mnist_width_height, hidden),
            torch.nn.ReLU(),
            MFLinear(hidden, hidden),
            torch.nn.ReLU(),
            MFLinear(hidden, mnist_classes),
            torch.nn.LogSoftmax(dim=2),
        ]
    )

    params = torch.hstack(
        [p.clone().reshape(-1) for p in mnist_net.parameters()]
    ).reshape(1, -1)
    params = torch.vstack([params, params + 0.01, params])

    batch = torch.rand(1, 7, mnist_width_height, mnist_width_height)

    out = mnist_param_net.forward(params, batch.repeat(3, 1, 1, 1))
    _out = mnist_net(batch[0])
    assert out[1][0].isclose(_out, atol=1e-5).all()
    assert not out[1][1].isclose(_out, atol=1e-5).all()
    assert out[1][2].isclose(_out, atol=1e-5).all()


def test_mnist_multimodel():
    mnist_width_height = 28
    mnist_classes = 10
    hidden = 32
    mnist_net = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(mnist_width_height * mnist_width_height, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, mnist_classes),
        torch.nn.LogSoftmax(dim=1),
    )

    mnist_param_net = param_nn_from_sequential(mnist_net)

    params = torch.hstack(
        [p.clone().reshape(-1) for p in mnist_net.parameters()]
    ).reshape(1, -1)
    params = torch.vstack([params, params + 0.01, params])

    batch = torch.rand(1, 7, mnist_width_height, mnist_width_height)

    out = mnist_param_net.forward(params, batch.repeat(3, 1, 1, 1))
    _out = mnist_net(batch[0])
    assert out[1][0].isclose(_out, atol=1e-5).all()
    assert not out[1][1].isclose(_out, atol=1e-5).all()
    assert out[1][2].isclose(_out, atol=1e-5).all()


def test_mnistcnn_multimodel():
    mnist_width_height = 28
    mnist_cnn = MNISTCNN()

    mnist_param_net = param_nn_from_sequential(mnist_cnn.net)

    params = torch.hstack(
        [p.clone().reshape(-1) for p in mnist_cnn.parameters()]
    ).reshape(1, -1)
    params = torch.vstack([params, params + 0.01, params])

    batch = torch.rand(1, 7, mnist_width_height, mnist_width_height)
    out = mnist_param_net.forward(params, batch.repeat(3, 1, 1, 1))
    _out = mnist_cnn(batch[0])
    assert out[1][0].isclose(_out, atol=1e-5).all()
    assert not out[1][1].isclose(_out, atol=1e-5).all()
    assert out[1][2].isclose(_out, atol=1e-5).all()
