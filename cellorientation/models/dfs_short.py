import torch.nn as nn
import torch

from .dfs_basic import Model as BaseModel


class Model(BaseModel):
    def __init__(self, n_in, activation="ReLU", w_init=1, w_thresh=1e-2):
        super(Model, self).__init__()

        self.w = nn.Parameter(torch.zeros(n_in).float())
        #         nn.init.constant_(self.w, w_init)

        nn.init.normal_(self.w, 0, 0.2)
        self.w = torch.abs(self.w)

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
