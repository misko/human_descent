import torch


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
        return (
            models_params[:, self.params :],
            torch.einsum("moi,mbi->mbo", _weights, x) + _bias,
        )
