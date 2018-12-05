import torch
from .. import SimpleLogger

from .. import basic_trainer

import os
import pickle


class Model(basic_trainer.Model):
    def __init__(
        self,
        net,
        opt,
        data_provider,
        crit_recon,
        gpu_ids,
        save_dir,
        n_epochs=300,
        save_state_iter=1,
        save_progress_iter=1,
    ):

        super(Model, self).__init__(
            data_provider,
            n_epochs,
            gpu_ids,
            save_dir,
            save_state_iter,
            save_progress_iter,
        )

        logger_path = "{}/logger.pkl".format(save_dir)

        if os.path.exists(logger_path):
            self.logger = pickle.load(open(logger_path, "rb"))
        else:
            print_str = (
                "[{epoch:%d}][{iter:%d}] reconLoss: {recon_loss:%.6f} time: {time:%.2f}"
            )

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

        recon_loss.backward()
        opt.step()

        log = {"recon_loss": recon_loss.item(), "z": z.cpu().numpy()}

        return log

    def save_progress(self):
        #         gpu_id = self.gpu_ids[0]
        #         epoch = self.get_current_epoch()

        #         data_provider = self.data_provider
        #         net = self.net
        pass

    def save(self):
        pass
