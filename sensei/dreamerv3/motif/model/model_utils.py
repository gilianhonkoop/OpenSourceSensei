from typing import Optional

import numpy as np
import torch
from numpy.core.multiarray import ndarray
from torch import nn


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def nonlinearity(nonlinearity_cfg):
    if nonlinearity_cfg == "elu":
        return nn.ELU(inplace=True)
    elif nonlinearity_cfg == "relu":
        return nn.ReLU(inplace=True)
    elif nonlinearity_cfg == "tanh":
        return nn.Tanh()
    elif nonlinearity_cfg == "none":
        return None
    else:
        raise Exception("Unknown nonlinearity")


class ConvEncoder(nn.Module):
    def __init__(
        self,
        activation,
        conv_filters,
        fc_layer_size,
        encoder_extra_fc_layers,
        input_shape=(3, 224, 224),
        downsample=False,
    ):
        super().__init__()
        conv_layers = []
        for layer in conv_filters:
            if layer == "maxpool_2x2":
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(nonlinearity(activation))
            else:
                raise NotImplementedError(f"Layer {layer} not supported!")
        if downsample:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, input_shape)

        fc_layers = []
        for i in range(encoder_extra_fc_layers):
            size = self.conv_head_out_size if i == 0 else fc_layer_size
            fc_layers.extend([nn.Linear(size, fc_layer_size), nonlinearity(activation)])

        self.fc_layers = nn.Sequential(*fc_layers)
        self.fc_layer_size = fc_layer_size

    def forward(self, input):
        x = self.conv_head(input)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.fc_layers(x)
        return x


def torch_clip(x, min_val, max_val):
    if min_val is None and max_val is None:
        raise ValueError("One of max or min must be given")
    elif min_val is None:
        return torch.min(x, max_val)
    elif max_val is None:
        return torch.max(x, min_val)
    else:
        return torch.max(torch.min(x, max_val), min_val)


class Normalizer:
    count: float
    sum_of_squares: ndarray
    sum: ndarray

    def __init__(self, shape, eps=1e-6, device=torch.device("cuda:0"), clip_range=(None, None)):
        self.mean = 0.0
        self.std = 1.0
        self.eps = eps
        self.shape = shape
        self.clip_range = clip_range
        self.device = device

        self.mean_tensor = torch.zeros(1).to(self.device)
        self.std_tensor = torch.ones(1).to(self.device)

        self.re_init()

    def re_init(self):
        self.sum = np.zeros(self.shape)
        self.sum_of_squares = np.zeros(self.shape)
        self.count = 1.0

    def update(self, data):
        self.sum += np.sum(data, axis=0)
        self.sum_of_squares += np.sum(np.square(data), axis=0)
        self.count += data.shape[0]

        self.mean = self.sum / self.count
        self.std = np.maximum(
            self.eps,
            np.sqrt(self.sum_of_squares / self.count - np.square(self.sum / self.count) + self.eps),
        )

        self.mean_tensor = torch.from_numpy(self.mean).float().to(self.device)
        self.std_tensor = torch.from_numpy(self.std).float().to(self.device)

    def normalize(self, data, out=None):
        if isinstance(data, torch.Tensor):
            if out is None:
                res = (data - self.mean_tensor) / self.std_tensor
                if not tuple(self.clip_range) == (None, None):
                    return torch_clip(res, *self.clip_range)
                else:
                    return res
            else:
                torch.sub(data, self.mean_tensor, out=out)
                torch.divide(out, self.std_tensor, out=out)
                if not tuple(self.clip_range) == (None, None):
                    torch.clip(out, min=self.clip_range[0], max=self.clip_range[1], out=out)
        else:
            res = (data - self.mean) / self.std
            if not tuple(self.clip_range) == (None, None):
                return np.clip(res, *self.clip_range)
            else:
                return res

    def denormalize(self, data, out=None):
        if isinstance(data, torch.Tensor):
            if out is None:
                return data * self.std_tensor + self.mean_tensor
            else:
                torch.multiply(data, self.std_tensor, out=out)
                torch.add(out, self.mean_tensor, out=out)
        else:
            return data * self.std + self.mean

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "sum": self.sum,
            "sum_of_squares": self.sum_of_squares,
            "count": self.count,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

        self.mean_tensor = torch.from_numpy(np.asarray(self.mean)).float().to(self.device)
        self.std_tensor = torch.from_numpy(np.asarray(self.std)).float().to(self.device)


def build_mlp(
    input_dim: int,
    output_dim: int,
    size: int,
    num_layers: int,
    activation: str,
    output_activation: Optional[str] = "none",
    layer_norm: bool = True,
):
    activation = nonlinearity(activation)
    output_activation = nonlinearity(output_activation)

    all_dims = [input_dim] + [size] * num_layers
    hidden_layers = []
    for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
        hidden_layers.append(nn.Linear(in_dim, out_dim))
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(size))
        if activation is not None:
            hidden_layers.append(activation)
    layers = hidden_layers + [nn.Linear(all_dims[-1], output_dim)]
    if output_activation is not None:
        layers.append(output_activation)

    layers = nn.Sequential(*layers)
    return layers


if __name__ == "__main__":

    input_ch = 3
    conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]

    model = ConvEncoder(
        activation="elu",
        conv_filters=conv_filters,
        fc_layer_size=512,
        encoder_extra_fc_layers=2,
        input_shape=(3, 224, 224),
        downsample=True,
    )
    print("Model! ", model)

    out = model(torch.zeros(1, input_ch, 224, 224))  # This will be the final logits over classes

    print("Model output shape with just convolutions! ", out.shape)
