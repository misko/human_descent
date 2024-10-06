import torch


class Sequential:
    def __init__(self, modules_list):
        self.params = 0
        self.modules_list = modules_list

    def forward(self, models_params, x):
        for module in self.modules_list:
            if isinstance(module, torch.nn.Module):
                x = module(x)
            else:
                models_params, x = module.forward(models_params, x)
        return models_params, x


class Linear:
    def __init__(self, input_channels, output_channels):
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.weight_params = self.input_channels * self.output_channels
        self.bias_params = self.output_channels
        self.params = self.weight_params + self.bias_params

    def forward(self, models_params, x):
        # model_params ~ (models, params)
        _weights = models_params[:, : self.weight_params].reshape(
            -1, self.output_channels, self.input_channels
        )
        _bias = models_params[
            :, self.weight_params : self.weight_params + self.bias_params
        ].reshape(-1, 1, self.bias_params)
        # x ~ (batch,input_dim)
        output = torch.einsum("moi,mbi->mbo", _weights, x) + _bias
        # print(
        #     "PARAMNN",
        #     _weights[0].abs().mean().item(),
        #     _bias[0].abs().mean().item(),
        #     output[0].abs().mean(),
        # )
        return (
            models_params[:, self.params :],
            output,
        )
