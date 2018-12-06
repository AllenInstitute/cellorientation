import torch
from .. import SimpleLogger
from . import ae
from ..utils import kl_divergence

import os
import pickle


class Model(ae.Model):
    def __init__(
        self,
        net,
        opt,
        n_epochs,
        gpu_ids,
        save_dir,
        data_provider,
        crit_recon,
        save_state_iter=1,
        save_progress_iter=1,
        beta=1,
        beta_start=1000,
        beta_iters_max=12500,
        c_max=500,
        c_iters_max=80000,
        gamma=500,
        objective="H",
        kld_avg=False,
    ):

        super(Model, self).__init__(
            net,
            opt,
            data_provider,
            crit_recon,
            gpu_ids,
            save_dir,
            n_epochs,
            save_state_iter,
            save_progress_iter,
        )

        self.beta = beta
        self.beta_start = beta_start
        self.beta_iters_max = beta_iters_max
        self.kld_avg = kld_avg
        self.objective = objective

        logger_path = "{}/logger.pkl".format(save_dir)

        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            print_str = "[{epoch:%d}][{iter:%d}] reconLoss: {recon_loss:%.6f} kld: {kld_loss:%.6f} total: {total_loss:%.6f} time: {time:%.2f}"

            self.logger = SimpleLogger(print_str)

    def iteration(self):

        torch.cuda.empty_cache()

        gpu_id = self.gpu_ids[0]

        net = self.net
        opt = self.opt
        crit_recon = self.crit_recon

        # do this just incase anything upstream changes these values
        net.train(True)

        opt.zero_grad()

        x = self.data_provider.next()
        x = x.cuda(gpu_id)

        #####################
        # train autoencoder
        #####################

        # Forward passes
        x_hat, z = net(x)

        recon_loss = crit_recon(x_hat, x)

        kld, _, _ = kl_divergence(z[0], z[1])
        if self.objective == "H":
            beta_vae_loss = recon_loss + self.beta * kld
        elif self.objective == "H_eps":
            beta_vae_loss = recon_loss + torch.abs((self.beta * kld) - x.shape[0] * 0.1)
        elif self.objective == "B":
            C = torch.clamp(
                torch.Tensor(
                    [self.c_max / self.c_iters_max * len(self.logger)]
                ).type_as(x),
                0,
                self.c_max,
            )
            beta_vae_loss = recon_loss + self.gamma * (kld - C).abs()

        beta_vae_loss.backward(retain_graph=True)
        opt.step()

        log = {
            "recon_loss": recon_loss.item(),
            "kld_loss": kld.item(),
            "total_loss": beta_vae_loss.item(),
            "z": [e.cpu().numpy() for e in z],
        }

        return log

    def save_progress(self):
        #         gpu_id = self.gpu_ids[0]
        #         epoch = self.get_current_epoch()

        #         data_provider = self.data_provider
        #         net = self.net
        pass
