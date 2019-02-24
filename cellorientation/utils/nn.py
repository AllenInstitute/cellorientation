import torch.nn as nn
from .utils import get_activation


class BasicLayer(nn.Module):
    def __init__(self, n_in, n_out, activation="ReLU"):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(n_in, n_out, bias=False),
            nn.BatchNorm1d(n_out),
            get_activation(activation),
        )

    def forward(self, x):
        return self.main(x)


class ResidualLayer1d(nn.Module):
    def __init__(
        self, ch_in, ch_hidden, activation="ReLU", bias=False, activation_last="ReLU"
    ):
        super(ResidualLayer1d, self).__init__()

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
