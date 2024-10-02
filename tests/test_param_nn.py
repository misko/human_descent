import torch

from hudes.param_nn import Linear


def test_linear():

    in_channels = 3
    out_channels = 5
    batch_size = 7
    n_models = 3
    model_params = torch.rand([n_models, in_channels * out_channels + out_channels])

    batch = torch.rand(1, batch_size, in_channels).expand(
        n_models, batch_size, in_channels
    )

    _mp, out = Linear(in_channels, out_channels).forward(model_params, batch)

    outputs = []
    for model_idx in range(n_models):
        _weight = model_params[model_idx, : in_channels * out_channels].reshape(
            out_channels, in_channels
        )
        _bias = model_params[model_idx, in_channels * out_channels :]
        layer = torch.nn.Linear(in_channels, out_channels)
        layer.weight.data.copy_(_weight)
        layer.bias.data.copy_(_bias)
        outputs.append(layer(batch[0]).unsqueeze(0))

    assert torch.vstack(outputs).isclose(out).all()
