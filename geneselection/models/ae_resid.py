import torch.nn as nn
from ..utils.utils import get_activation


class BasicLayer(nn.Module):
    def __init__(self, n_in, n_out, activation="ReLU", bias=False):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(n_in, n_out, bias=bias),
            nn.BatchNorm1d(n_out),
            get_activation(activation),
        )

    def forward(self, x):
        return self.main(x)


class ResidualLayer1d(nn.Module):
    def __init__(
        self, ch_in, ch_hidden, activation="ReLU", bias=False, activation_last=None
    ):
        super(ResidualLayer1d, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.resid = nn.Sequential(
            nn.Linear(ch_in, ch_hidden, bias=bias),
            nn.BatchNorm1d(ch_hidden),
            get_activation(activation),
            nn.Linear(ch_hidden, ch_in, bias=bias),
            nn.BatchNorm1d(ch_in),
        )

        self.activation = get_activation(activation_last)

    def forward(self, x):
        x = x + self.resid(x)
        x = self.activation(x)

        return x


class ResidualBlock1d(nn.Module):
    def __init__(self, ch_in, ch_hidden, n_layers, activation="ReLU", bias=False):
        super(ResidualBlock1d, self).__init__()

        self.module_list = nn.ModuleList()

        for i in range(n_layers):
            self.module_list.append(ResidualLayer1d(ch_in, ch_hidden, activation, bias))

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)

        return x


class Enc(nn.Module):
    def __init__(self, n_in, layer_scaling=[8, 16, 32, 64, 128], activation="RelU"):
        super(Enc, self).__init__()

        self.module_list = nn.ModuleList(
            [BasicLayer(n_in, int(n_in / layer_scaling[0]))]
        )

        for i in range(len(layer_scaling) - 1):
            layer = layer_scaling[i]
            layer_out = layer_scaling[i + 1]

            self.module_list.append(
                ResidualBlock1d(int(n_in / layer), int(n_in / (layer * 4)), layer)
            )

            self.module_list.append(
                BasicLayer(int(n_in / layer), int(n_in / layer_out))
            )

        self.module_list.append(
            nn.Linear(int(n_in / layer_out), int(n_in / layer_out), bias=False)
        )

    def forward(self, x):

        for layer in self.module_list:
            x = layer(x)

        return x


class Dec(nn.Module):
    def __init__(self, n_out, layer_scaling=[128, 64, 32, 16, 8], activation="RelU"):
        super(Dec, self).__init__()

        self.module_list = nn.ModuleList()

        for i in range(len(layer_scaling) - 1):
            layer = layer_scaling[i]
            layer_out = layer_scaling[i + 1]

            self.module_list.append(
                ResidualBlock1d(int(n_out / layer), int(n_out / (layer * 4)), layer)
            )

            self.module_list.append(
                BasicLayer(int(n_out / layer), int(n_out / layer_out))
            )

        self.module_list.append(
            nn.Linear(int(n_out / layer_out), int(n_out), bias=False)
        )

    def forward(self, x):

        for layer in self.module_list:
            x = layer(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, n_in, n_latent=512, activation="ReLU"):
        super(Autoencoder, self).__init__()

        self.enc = Enc(n_in, activation=activation)
        self.dec = Dec(n_in, activation=activation)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z
