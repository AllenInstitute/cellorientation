import torch.nn as nn
from ..utils import get_activation


class BasicLayer(nn.Module):
    def __init__(self, n_in, n_out, activation="ReLU"):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            nn.linear(n_in, n_out, bias=False),
            nn.BatchNorm1d(n_out),
            get_activation(activation),
        )

    def forward(self, x):
        return self.main(x)


class Enc(nn.Module):
    def __init__(self, n_in, n_latent=512, activation="RelU"):
        super(Enc, self).__init__()

        self.main = nn.Sequential(
            BasicLayer(n_in, 1024),
            BasicLayer(1024, 512),
            nn.linear(512, n_latent, bias=False),
        )

    def forward(self, x):
        return self.main(x)


class Dec(nn.Module):
    def __init__(self, n_out, n_latent=512, activaiton="ReLU"):
        super(Dec, self).__init__()

        self.main = nn.Sequential(
            BasicLayer(n_latent, 512),
            BasicLayer(512, 1024),
            nn.linear(1024, n_out, bias=False),
        )

    def forward(self, x):
        return self.main(x)


class AutoEncoder(nn.Module):
    def __init__(self, n_in, n_latent=512, activation="ReLU"):
        super(AutoEncoder, self).__init__()

        self.enc = Enc(n_in, n_latent, activation=activation)
        self.dec = Dec(n_in, n_latent, activation=activation)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z