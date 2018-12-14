import importlib
import json
import os
import sys
import torch
import torch.optim
import geneselection.utils as utils
from geneselection.datasets.dataset import gsdataset_from_anndata
import numpy as np


def run(
    network_kwargs,
    optim_kwargs,
    trainer_kwargs,
    dataset_kwargs,
    data_loader_kwargs,
    loss_kwargs,
    save_dir,
    gpu_ids,
    seed=0,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in gpu_ids])
    gpu_ids = list(range(0, len(gpu_ids)))

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if len(gpu_ids) == 1:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # load the dataloader
    anndataset_module = importlib.import_module(dataset_kwargs["name"])
    anndata = anndataset_module.load(**dataset_kwargs["kwargs"])

    _, anndata_splits = utils.data.split_anndata(
        anndata, test_size=0.2, random_state=seed
    )

    ds_train = gsdataset_from_anndata(anndata_splits["train"])
    ds_validate = gsdataset_from_anndata(anndata_splits["test"])

    # train split
    data_loader_module, data_loader_name = data_loader_kwargs["name"].rsplit(".", 1)
    data_loader_module = importlib.import_module(data_loader_module)
    dataloader = getattr(data_loader_module, data_loader_name)
    dataloader_train = dataloader(ds_train, **data_loader_kwargs["kwargs"])

    # validate split
    data_loader_kwargs["kwargs"]["shuffle"] = False
    data_loader_kwargs["kwargs"]["drop_last"] = True

    dataloader_validate = dataloader(ds_validate, **data_loader_kwargs["kwargs"])

    # load the networks
    network_module, network_name = network_kwargs["name"].rsplit(".", 1)
    network_module = importlib.import_module(network_module)
    network = getattr(network_module, network_name)(**network_kwargs["kwargs"])

    network = network.cuda(gpu_ids[0])

    # load the optimizer
    optim_module, optim_name = optim_kwargs["name"].rsplit(".", 1)
    optim_module = importlib.import_module(optim_module)
    opt = getattr(optim_module, optim_name)(
        params=network.parameters(), **optim_kwargs["kwargs"]
    )

    # load the loss functions
    loss_module, loss_name = loss_kwargs["name"].rsplit(".", 1)
    loss_module = importlib.import_module(loss_module)
    loss = getattr(loss_module, loss_name)(**loss_kwargs["kwargs"])

    # load the trainer model
    trainer_module = importlib.import_module(trainer_kwargs["name"])
    trainer = trainer_module.Model(
        dataloader_train=dataloader_train,
        dataloader_validate=dataloader_validate,
        net=network,
        opt=opt,
        loss=loss,
        save_dir=save_dir,
        gpu_ids=gpu_ids,
        **trainer_kwargs["kwargs"]
    )

    # start training
    trainer.train()


if __name__ == "__main__":

    json_input = sys.argv[1]

    with open(json_input, "rb") as f:
        kwargs = json.load(f)

    # run it
    run(**kwargs)
