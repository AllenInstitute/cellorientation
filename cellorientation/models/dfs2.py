import torch.nn as nn
import torch
from geneselection.utils.nn import ResidualLayer1d


class Model(nn.Module):
    def __init__(self, n_in, activation="ReLU", w_init=1):
        super(Model, self).__init__()

        n32 = 2048

        self.w = nn.Parameter(torch.zeros(n_in).float())
        nn.init.constant_(self.w, w_init)

        self.main = nn.Sequential(
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32),
            ResidualLayer1d(n_in, n32, activation_last=None),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 0, 0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.main.apply(weights_init)

    def forward(self, x):

        # x = torch.cat(x, 1)

        x = x.mul(self.w)

        x = self.main(x)

        return x
