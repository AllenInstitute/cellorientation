import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):

        if self.training:
            x = x.mul(self.w)
        else:
            x = x.mul(self.w * self.get_w_mask())

        x = self.main(x)

        return x

    def get_w_mask(self):
        self.w_thresh = self.w_thresh.type_as(self.w)

        return (self.w >= self.w_thresh).float()
