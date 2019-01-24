import torch.nn as nn
import torch


class BasicLayer(nn.Module):
    def __init__(self, n_in, n_out, actvation="ReLU", bias=False):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(n_in, n_out, bias=bias),
            nn.BatchNorm1d(n_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main(x)

        return x


class Model(nn.Module):
    def __init__(self, n_in, activation="ReLU", w_init=1, w_thresh=1e-2):
        super(Model, self).__init__()

        self.w = nn.Parameter(torch.zeros(n_in).float())
        nn.init.constant_(self.w, w_init)

        self.w_thresh = torch.Tensor([w_thresh])

        self.main = nn.Sequential(nn.Linear(n_in, n_in))

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

        if self.training:
            x = x.mul(self.w)
        else:
            self.w_thresh = self.w_thresh.type_as(x)
            x = x.mul(self.w * (self.w >= self.w_thresh).float())

        x = self.main(x)

        return x
