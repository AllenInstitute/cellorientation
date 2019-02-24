import torch.nn as nn
import torch

from .dfs_basic import Model as BaseModel


class Model(BaseModel):
    def __init__(self, n_in, activation="ReLU", w_init=0, w_thresh=1e-2):
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

        if self.training:
            x = x.mul(self.get_w())
        else:
            x = x.mul(self.get_w() * self.get_w_mask())

        x = self.main(x)

        return x

    def get_w(self):
        return torch.nn.Sigmoid()(self.w)

    def get_w_mask(self):
        self.w_thresh = self.w_thresh.type_as(self.w)

        return (self.get_w() >= self.w_thresh).float()
