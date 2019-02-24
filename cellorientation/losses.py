import torch


class gaussian_aleotoric(torch.nn.Module):
    def __init__(self):
        super(gaussian_aleotoric, self).__init__()

    def forward(self, x_mu, sig_log_var, target, mask=None):

        diff = 0.5 * torch.exp(-sig_log_var) * (target - x_mu) ** 2 + 0.5 * (
            sig_log_var
        )

        if mask is None:
            mask = torch.tensor(target.shape).type_as(target)

        err = torch.mean(torch.masked_select(diff, mask > 0))

        return err


class MSELoss_masked(torch.nn.Module):
    def __init__(self):
        super(MSELoss_masked, self).__init__()

        self.loss = torch.nn.MSELoss()

    def forward(self, x_hat, x_target, mask=None):

        if mask is None:
            mask = torch.ones(x_target.shape).type_as(x_target)

        x_hat_masked = torch.masked_select(x_hat, mask > 0)
        x_target_masked = torch.masked_select(x_target, mask > 0)

        if len(x_hat_masked) == 0:
            err = torch.zeros(1).type_as(x_target)
        else:
            err = self.loss(x_hat_masked, x_target_masked)

        return err
