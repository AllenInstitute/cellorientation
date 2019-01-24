import numpy as np
import time
import os
from ..simplelogger import SimpleLogger
import pickle
import shutil

import torch
from .. import utils
from ..utils import plots

# This is the base class for trainers


class Model(object):
    def __init__(
        self,
        dataloader,
        n_epochs,
        gpu_ids,
        save_dir,
        save_state_iter=1,
        save_progress_iter=1,
        provide_decoder_vars=0,
    ):

        # self.__dict__.update(kwargs)

        self.dataloader = dataloader
        self.n_epochs = n_epochs

        self.gpu_ids = gpu_ids

        self.save_dir = save_dir

        self.save_state_iter = save_state_iter
        self.save_progress_iter = save_progress_iter

        self.provide_decoder_vars = provide_decoder_vars

        self.iters_per_epoch = np.ceil(len(dataloader.dataset) / dataloader.batch_size)

        logger_path = "{}/logger.pkl".format(save_dir)

        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            print_str = "[{epoch:d}][{iter:d}] reconLoss: {recon_loss:.8f} validLoss: {valid_loss:.8f} time: {time:.2f}"

            self.logger = SimpleLogger(print_str)

    def get_current_iter(self):
        return int(len(self.logger))

    def get_current_epoch(self, iteration=-1):
        if iteration == -1:
            iteration = self.get_current_iter()

        return int(np.floor(iteration / self.iters_per_epoch))

    def save(self, save_dir):
        save_dir = self.save_dir
        gpu_id = self.gpu_ids[0]

        n_iters = self.get_current_iter()

        net_save_path = "{0}/net.pth".format(save_dir)
        net_save_path_final = "{0}/net_{1}.pth".format(save_dir, n_iters)

        utils.utils.save_state(self.net, self.opt, net_save_path, gpu_id)
        shutil.copyfile(net_save_path, net_save_path_final)

        pickle.dump(self.logger, open("{0}/logger.pkl".format(save_dir), "wb"))

    def save_progress(self):
        # History
        plots.history(
            self.logger,
            "{0}/history.png".format(self.save_dir),
            loss_ax1=["recon_loss", "valid_loss"],
        )
        # Short History

        history_len = int(len(self.logger) / 2)

        if history_len > 10000:
            history_len = 10000

        plots.history(
            self.logger,
            "{0}/history_short.png".format(self.save_dir),
            loss_ax1=["recon_loss", "valid_loss"],
            history_len=history_len,
        )

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
        y = mb["Y"].float().cuda(gpu_id)

        opt.zero_grad()

        y_hat = net(x)
        recon_loss = loss(y_hat, y)

        total_loss = recon_loss
        total_loss.backward()
        opt.step()

        # Validation results
        _, mb = next(enumerate(self.dataloader_validate))
        x = mb["X"].float().cuda(gpu_id)
        y = mb["Y"].float().cuda(gpu_id)

        net.train(False)
        with torch.no_grad():
            y_hat = net(x)

        valid_loss = loss(y_hat, y)

        log = {"recon_loss": recon_loss.item(), "valid_loss": valid_loss.item()}

        return log

    def maybe_save(self):

        epoch = self.get_current_epoch(self.get_current_iter() - 1)
        epoch_next = self.get_current_epoch(self.get_current_iter())

        saved = False
        if epoch != epoch_next:
            if (epoch_next % self.save_progress_iter) == 0:
                print("saving progress")
                self.save_progress()

            if (epoch_next % self.save_state_iter) == 0:
                print("saving state")
                self.save(self.save_dir)

            saved = True

        return saved

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
