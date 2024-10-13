from typing import Tuple

import torch


def get_param_module(module):
    if isinstance(module, torch.nn.Flatten):
        return torch.nn.Flatten(start_dim=module.start_dim + 1)
    elif isinstance(module, torch.nn.LogSoftmax):
        return torch.nn.LogSoftmax(dim=module.dim + 1)
    elif isinstance(module, torch.nn.MaxPool2d):
        return MFMaxPool2d(
            kernel_size=module.kernel_size, stride=module.stride, padding=module.padding
        )
    elif isinstance(module, Unsqueeze):
        return Unsqueeze(module.dim + 1)
    elif not hasattr(module, "parameters") or len(list(module.parameters())) == 0:
        return module
    elif isinstance(module, torch.nn.Sequential):
        return param_nn_from_sequential(module)
    elif isinstance(module, torch.nn.Conv2d):
        return MFConv2d(
            input_channels=module.in_channels,
            output_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
        )
    elif isinstance(module, torch.nn.Linear):
        out_channels, in_channels = module.weight.shape
        return MFLinear(input_channels=in_channels, output_channels=out_channels)
    return torch.nn.Identity()


def param_nn_from_sequential(model):
    return MFSequential([get_param_module(m) for m in model])


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class MFSequential:
    def __init__(self, modules_list):
        self.params = 0
        self.modules_list = modules_list

    def forward(self, models_params, x):
        for module in self.modules_list:
            print("BEFORE", models_params.shape, x.shape)
            if isinstance(module, torch.nn.Module):
                print("RUNNING MODULE")
                x = module(x)
            else:
                models_params, x = module.forward(models_params, x)
            print("AFTER", models_params.shape, x.shape)
        return models_params, x


class MFMaxPool2d:
    def __init__(self, kernel_size, stride, padding):
        self.maxpool2d = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, models_params, x):
        n, k, out_c, in_h, in_w = x.shape
        output = self.maxpool2d(x.reshape(n * k, out_c, in_h, in_w))
        _, _out_c, out_h, out_w = output.shape
        output = output.reshape(n, k, out_c, out_h, out_w)

        return (
            models_params,
            output,
        )


@torch.compile
class MFConv2d:
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_params = (
            self.input_channels
            * self.output_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        self.bias_params = self.output_channels
        self.params = self.weight_params + self.bias_params

    def forward(self, models_params: torch.Tensor, x: torch.Tensor):
        # model_params ~ (models, params)
        weights = models_params[:, : self.weight_params].reshape(
            -1,
            self.output_channels,
            self.input_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        bias = models_params[
            :, self.weight_params : self.weight_params + self.bias_params
        ].reshape(-1, self.bias_params)

        n, out_c, in_c, kh, kw = weights.shape
        _n, im_k, im_in_c, im_h, im_w = x.shape
        assert n == _n

        # x is shape (n, k,in_c,h.,w.)
        # but we need (k,n*in_c,h.,w.)
        _x = x.transpose(0, 1).reshape(im_k, n * im_in_c, im_h, im_w)

        # weights are (n, out_c, in_c, kh, kw)
        # but we need (n*out_c, in_c, kh, kw)
        _weights = weights.reshape(n * out_c, in_c, kh, kw)

        # bias is (n,out_c)
        # we need (n*out_c)
        _bias = bias.reshape(-1)

        output = torch.nn.functional.conv2d(
            _x, _weights, _bias, stride=self.stride, padding=self.padding, groups=n
        )
        _, _, out_h, out_w = output.shape
        output = output.reshape(im_k, n, out_c, out_h, out_w).transpose(0, 1)

        return (
            models_params[:, self.params :],
            output,
        )


@torch.compile
class MFLinear:
    def __init__(self, input_channels: int, output_channels: int):
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.weight_params = self.input_channels * self.output_channels
        self.bias_params = self.output_channels
        self.params = self.weight_params + self.bias_params

    # @torch.compile
    def forward(self, models_params: torch.Tensor, x: torch.Tensor):
        # model_params ~ (models, params)
        _weights = models_params[:, : self.weight_params].reshape(
            -1, self.output_channels, self.input_channels
        )
        _bias = models_params[
            :, self.weight_params : self.weight_params + self.bias_params
        ].reshape(-1, 1, self.bias_params)
        # x ~ (batch,input_dim)
        output = torch.einsum("moi,mbi->mbo", _weights, x) + _bias
        return (
            models_params[:, self.params :],
            output,
        )
