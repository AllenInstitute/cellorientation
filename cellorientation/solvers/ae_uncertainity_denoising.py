import torch
from ..simplelogger import SimpleLogger

from . import basic_net_trainer
from .. import utils
from ..utils import plots

import os
import pickle
import shutil


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

        logger_path = "{}/logger.pkl".format(save_dir)

        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            print_str = (
                "[{epoch:d}][{iter:d}] trainLoss: {train_loss:.8f} validLoss: {valid_loss:.8f} time: {time:.2f}"
                # "[{epoch:d}][{iter:d}] trainLoss: {train_loss:.8f} time: {time:.2f}"
            )

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
        x = mb["X"].cuda(gpu_id)

        # make a mask
        dropout_rate = torch.zeros(x.shape[0]).uniform_()
        mask = torch.zeros(x.shape).cuda(gpu_id)
        for i, d in enumerate(dropout_rate):
            mask[i] = mask[i].bernoulli_(d)

        opt.zero_grad()
        x_mu, x_var = net((x * mask).detach())
        train_loss = loss(x_mu, x_var, x, mask=mask == 0)
        train_loss.backward()
        opt.step()

        # Validation results
        _, mb = next(enumerate(self.dataloader_validate))
        x = mb["X"].cuda(gpu_id)

        # make a mask
        dropout_rate = torch.zeros(x.shape[0]).uniform_()
        mask = torch.zeros(x.shape).cuda(gpu_id)
        for i, d in enumerate(dropout_rate):
            mask[i] = mask[i].bernoulli_(d)

        net.train(False)
        with torch.no_grad():
            x_mu, x_var = net((x * mask).detach())

        valid_loss = loss(x_mu, x_var, x, mask=mask == 0)

        log = {"train_loss": train_loss.item(), "valid_loss": valid_loss.item()}

        return log

    def save_progress(self):
        # History
        plots.history(
            self.logger,
            "{0}/history.png".format(self.save_dir),
            loss_ax1=["train_loss", "valid_loss"],
        )
        # Short History

        history_len = int(len(self.logger) / 2)

        if history_len > 10000:
            history_len = 10000

        plots.history(
            self.logger,
            "{0}/history_short.png".format(self.save_dir),
            loss_ax1=["train_loss", "valid_loss"],
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
