import torch

def run(model, optimizer, crit_loss, train_loader, val_loader, n_epochs, gpu_ids, save_dir, save_progress_iter=1, save_state_iter=10, print_prefix=""):

    if gpu_ids is not None:
        device = "cuda"
    else:
        device = "cpu"

    def step(engine, batch):
        x = batch["x"].to(device)
        
        model.zero_grad()

        x_hat, z_mask = net(x)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        
        loss = crit_loss(x_hat, x)

        loss.backward()
        opt_enc.step()
        opt_dec.step()        
        
        batch.pop("x", None)

        return {
            "loss": loss.item(),
            "z": z,
            **batch,
        }

    def evaluate(engine, batch):
        x = batch["x"].to(device)

        model.train(False)

        with torch.no_grad():
            z = enc(x)
            x_hat = dec(z)

        loss = crit_loss(x_hat, x)
        
        model.train(True)

        batch.pop("x", None)

        return {"loss": loss.item(), "z": z, **batch}

    evaluator = Engine(evaluate)

    @evaluator.on(Events.EPOCH_STARTED)
    def reset_val_stats(engine):
        engine.state.history = {}

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_val_stats(engine):
        for k in engine.state.output:
            if k not in engine.state.history:
                engine.state.history[k] = list()
            engine.state.history[k] += [engine.state.output[k]]

    trainer = Engine(step)
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % save_progress_iter == 0:
            print(
                "{}Epoch[{}] Iter[{}/{}] Loss:{:.4f} F1:{:.4f} Time:{:.2f}"
                "".format(
                    print_prefix,
                    engine.state.epoch,
                    iter,
                    len(train_loader),
                    engine.state.output["loss"],
                    engine.state.output["f1"],
                    timer.value(),
                )
            )
            
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine):
        if engine.state.epoch % save_state_iter == 0:
            model_utils.save_state(
                enc, opt_enc, "{0}/enc.pth".format(save_dir), gpu_ids[0]
            )
            model_utils.save_state(
                model,
                optimizer,
                "{0}/classifier_{1}.pth".format(save_dir, engine.state.epoch),
                gpu_ids[0],
            )

    @trainer.on(Events.EPOCH_STARTED)
    def reset_train_stats(engine):
        engine.state.history = {}

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_train_stats(engine):
        for k in engine.state.output:
            if k not in engine.state.history:
                engine.state.history[k] = list()
            engine.state.history[k] += [engine.state.output[k]]

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_train_preds(engine):
        metrics = engine.state.history

        for k in metrics:
            if isinstance(metrics[k][0], np.ndarray):
                metrics[k] = np.vstack(metrics[k]).tolist()

        metrics["f1_score"] = f1_score(
            np.array(metrics["targets"]),
            np.array(metrics["scores"]) > 0,
            average="macro",
        )

        save_path = "{0}/train_stats_{1}.json".format(save_dir, trainer.state.epoch)

        with open(save_path, "w") as outfile:
            json.dump(metrics, outfile)

        metrics.pop("targets", None)
        metrics.pop("scores", None)

        save_path = "{0}/train_stats_short_{1}.json".format(
            save_dir, trainer.state.epoch
        )
        with open(save_path, "w") as outfile:
            json.dump(metrics, outfile)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_val_preds(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.history

        for k in metrics:
            if isinstance(metrics[k][0], np.ndarray):
                metrics[k] = np.vstack(metrics[k]).tolist()

        metrics["f1_score"] = f1_score(
            np.array(metrics["targets"]),
            np.array(metrics["scores"]) > 0,
            average="macro",
        )
        save_path = "{0}/val_stats_{1}.json".format(save_dir, trainer.state.epoch)
        with open(save_path, "w") as outfile:
            json.dump(metrics, outfile)

        metrics.pop("targets", None)
        metrics.pop("scores", None)

        save_path = "{0}/val_stats_short_{1}.json".format(save_dir, trainer.state.epoch)
        with open(save_path, "w") as outfile:
            json.dump(metrics, outfile)

   
    trainer.run(train_loader, max_epochs=n_epochs)