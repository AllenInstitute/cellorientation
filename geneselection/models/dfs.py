import torch.nn as nn
import torch
from geneselection.utils.nn import ResidualLayer1d


class Model(nn.Module):
    def __init__(self, n_in, activation="ReLU"):
        super(Model, self).__init__()

        n32 = int(n_in / 32)

        self.w = nn.Parameter(torch.zeros(n_in).float())

        nn.init.constant_(self.w, 1)

        self.main = nn.Sequential(
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32, activation_last=None),
        )

    def forward(self, x):

        # x = torch.cat(x, 1)

        x = x.mul(self.w)

        x = self.main(x)

        return x
