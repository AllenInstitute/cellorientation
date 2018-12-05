import importlib
import json
import os
import sys


def run(
    network_kwargs,
    optim_kwargs,
    trainer_kwargs,
    data_provider_kwargs,
    loss_kwargs,
    save_dir,
    seed=0,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load the data_provider
    data_provider_module = importlib.import_module(data_provider_kwargs["name"])
    data_provider = data_provider_module.load(**data_provider_kwargs["kwargs"])

    # load the networks
    network_module = importlib.import_module(network_kwargs["name"])
    network = network_module(**network_kwargs["kwargs"])

    # load the optimizer
    optimizer_module = importlib.import_module(optim_kwargs["name"])
    optim = optimizer_module(network.parameters(), **optim_kwargs["kwargs"])

    # load the loss functions
    loss_module = importlib.import_module(loss_kwargs["name"])
    loss = loss_module(**loss_kwargs["kwargs"])

    # load the trainer model
    trainer_module = importlib.import_module(trainer_kwargs["name"])
    trainer = trainer_module(
        data_provider=data_provider,
        net=network,
        optim=optim,
        loss=loss,
        save_dir=save_dir,
        **trainer_kwargs["kwargs"]
    )

    trainer.train()


if __name__ == "__main__":

    json_input = sys.argv[1]

    with open(json_input, "rb") as f:
        kwargs = json.load(f)

    # run it
    run(**kwargs)
