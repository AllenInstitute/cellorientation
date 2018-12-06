import numpy as np
import time

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

        self.zAll = list()

    def get_current_iter(self):
        return int(len(self.logger))

    def get_current_epoch(self, iteration=-1):
        if iteration == -1:
            iteration = self.get_current_iter()

        return int(np.floor(iteration / self.iters_per_epoch))

    def save(self, save_dir):
        raise NotImplementedError

    def save_progress(self):
        raise NotImplementedError

    def iteration(self):
        raise NotImplementedError

    def maybe_save(self):

        epoch = self.get_current_epoch(self.get_current_iter() - 1)
        epoch_next = self.get_current_epoch(self.get_current_iter())

        saved = False
        if epoch != epoch_next and (
            (epoch_next % self.save_state_iter) == 0
            or (epoch_next % self.save_state_iter) == 0
        ):
            if (epoch_next % self.save_progress_iter) == 0:
                print("saving progress")
                self.save_progress()

            if (epoch_next % self.save_progress_iter) == 0:
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
