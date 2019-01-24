import torch
from ..simplelogger import SimpleLogger

from . import basic_net_trainer
from .. import utils
from ..utils import plots

import os
import pickle
import shutil

import numpy as np
import time


class Model(basic_net_trainer.Model):
    def __init__(
        self,
        net,
        opt,
        dataloader_train,
        dataloader_validate,
        loss,
        gpu_ids,
        save_dir,
        n_epochs=300,
        save_state_iter=1,
        save_progress_iter=1,
        lambda1=0.05,
        lambda2=1,
        alpha1=1e-4,
        alpha2=0,
    ):

        super(Model, self).__init__(
            dataloader=dataloader_train,
            n_epochs=n_epochs,
            gpu_ids=gpu_ids,
            save_dir=save_dir,
            save_state_iter=save_state_iter,
            save_progress_iter=save_progress_iter,
        )

        self.dataloader_validate = dataloader_validate
        self.net = net
        self.opt = opt
        self.loss = loss

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        logger_path = "{}/logger.pkl".format(save_dir)

        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            print_str = "[{epoch:d}][{iter:d}] lambaLoss: {lambda_loss:0.8f} alphaLoss {alpha_loss:.8f} |w|: {w_sum:.4f} reconLoss: {recon_loss:.8f} validLoss: {valid_loss:.8f} time: {time:.2f}"

            self.logger = SimpleLogger(print_str)

    def iteration(self):

        torch.cuda.empty_cache()

        gpu_id = self.gpu_ids[0]

        net = self.net
        opt = self.opt
        loss = self.loss

        # do this just incase anything upstream changes these values
        net.train(True)

        _, mb = next(enumerate(self.dataloader))
        x = mb["X"].float().cuda(gpu_id)

        opt.zero_grad()

        x_hat = net(x)
        recon_loss = loss(x_hat, x)

        lambda_loss = self.lambda1 * (
            (1 - self.lambda2) / 2 * torch.norm(net.w, 2)
            + self.lambda2 * torch.norm(net.w, 1)
        )

        weights = [
            module.weight
            for module in net.main.modules()
            if module.__class__.__name__ == "Linear"
        ]

        l2s = torch.sum(torch.stack([torch.norm(w, 2) for w in weights]))
        l1s = torch.sum(torch.stack([torch.norm(w, 1) for w in weights]))
        alpha_loss = self.alpha1 * ((1 - self.alpha2) / 2 * l2s + self.alpha2 * l1s)

        total_loss = recon_loss + lambda_loss + alpha_loss
        total_loss.backward()
        opt.step()

        # Validation results
        _, mb = next(enumerate(self.dataloader_validate))
        x = mb["X"].float().cuda(gpu_id)

        net.train(False)
        with torch.no_grad():
            x_hat = net(x)

        valid_loss = loss(x_hat, x)

        log = {
            "recon_loss": recon_loss.item(),
            "lambda_loss": lambda_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "valid_loss": valid_loss.item(),
            "weights_lambda": net.w.detach().cpu().numpy(),
            "w_sum": np.sum(np.abs(net.w.detach().cpu().numpy())),
        }

        return log

    def save_progress(self):
        # History
        plots.history(
            self.logger,
            "{0}/history.png".format(self.save_dir),
            loss_ax1=["recon_loss", "valid_loss"],
            loss_ax2=["lambda_loss", "alpha_loss"],
        )
        # Short History

        history_len = int(len(self.logger) / 2)

        if history_len > 10000:
            history_len = 10000

        plots.history(
            self.logger,
            "{0}/history_short.png".format(self.save_dir),
            loss_ax1=["recon_loss", "valid_loss"],
            loss_ax2=["lambda_loss", "alpha_loss"],
            history_len=history_len,
        )

    def save(self, save_dir):
        #         for saving and loading see:
        #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        save_dir = self.save_dir
        gpu_id = self.gpu_ids[0]

        n_iters = self.get_current_iter()

        net_save_path = "{0}/net.pth".format(save_dir)
        net_save_path_final = "{0}/net_{1}.pth".format(save_dir, n_iters)

        utils.utils.save_state(self.net, self.opt, net_save_path, gpu_id)
        shutil.copyfile(net_save_path, net_save_path_final)

        pickle.dump(self.logger, open("{0}/logger.pkl".format(save_dir), "wb"))

    def get_errors(self):
        net = self.net
        loss = self.loss
        gpu_id = self.gpu_ids[0]

        net.train(False)

        save_out = {}
        data_loader_names = ["train", "validate"]
        data_loaders = [self.dataloader, self.dataloader_validate]

        for name, loader in zip(data_loader_names, data_loaders):
            index = []
            x_hats = []
            losses = []

            for _, mb in enumerate(loader):

                x = mb["X"].float().cuda(gpu_id)

                with torch.no_grad():
                    x_hat = net(x)

                batch_loss = loss(x_hat, x)

                index += [mb["idx"].numpy()]
                x_hats += [x_hat.detach().cpu().numpy()]
                losses += [batch_loss.item()]

            save_out[name] = {}
            save_out[name]["index"] = index
            save_out[name]["x_hats"] = x_hats
            save_out[name]["losses"] = losses

        pickle.dump(save_out, open("{0}/errors.pkl".format(self.save_dir), "wb"))

    def train(self):
        start_iter = self.get_current_iter()

        for this_iter in range(
            int(start_iter), int(np.ceil(self.iters_per_epoch) * self.n_epochs)
        ):

            start = time.time()

            log = self.iteration()

            stop = time.time()
            deltaT = stop - start

            log["epoch"] = self.get_current_epoch()
            log["iter"] = self.get_current_iter()
            log["time"] = deltaT

            self.logger.add(log)
            self.maybe_save()

        self.get_errors()
