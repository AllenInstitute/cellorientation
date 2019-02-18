import json
import os
import importlib
import argparse
import datetime
import subprocess
import numpy as np
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None, help="gpu id")
    parser.add_argument(
        "--kwargs_path", type=str, default=None, help="kwargs for the classifier"
    )

    args = vars(parser.parse_args())

    nvals = 1001
    search_params = {}
    #     search_params["lambda1_vals"] = [1E-2, 5E-3, 1E-3, 5E-4, 1E-4, 5E-5, 1E-5, 5E-6, 1E-6,]
    search_params["lambda1_vals"] = list(np.linspace(0, 1e-2, 10000))
    search_params["lambda2_vals"] = [1]

    #     search_params["alpha1_vals"] = [1E-3, 5E-4, 1E-4, 5E-5, 1E-5]
    search_params["alpha1_vals"] = [1e-4]
    search_params["alpha2_vals"] = [0]

    # save the search params to the parent directory
    with open(args["kwargs_path"], "rb") as f:
        kwargs = json.load(f)

    if not os.path.exists(kwargs["save_parent"]):
        os.makedirs(kwargs["save_parent"])

    args_path = "{0}/search_params.json".format(kwargs["save_parent"])
    with open(args_path, "w") as f:
        json.dump(search_params, f, indent=4, sort_keys=True)

    for i in range(1000):

        np.random.seed(int(time.time()))

        lambda1 = float(np.random.choice(search_params["lambda1_vals"]))
        lambda2 = float(np.random.choice(search_params["lambda2_vals"]))
        alpha1 = float(np.random.choice(search_params["alpha1_vals"]))
        alpha2 = float(np.random.choice(search_params["alpha2_vals"]))

        if os.path.exists(args["kwargs_path"]):
            with open(args["kwargs_path"], "rb") as f:
                kwargs = json.load(f)

        kwargs["kwargs_path"] = args["kwargs_path"]
        kwargs["gpu_ids"] = args["gpu_ids"]

        kwargs["trainer_kwargs"]["kwargs"]["lambda1"] = lambda1
        kwargs["trainer_kwargs"]["kwargs"]["lambda2"] = lambda2
        kwargs["trainer_kwargs"]["kwargs"]["alpha1"] = alpha1
        kwargs["trainer_kwargs"]["kwargs"]["alpha2"] = alpha2

        the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        kwargs["save_dir"] = os.path.join(kwargs["save_parent"], the_time)
        kwargs.pop("save_parent")

        if not os.path.exists(kwargs["save_dir"]):
            os.makedirs(kwargs["save_dir"])

        kwargs["git_commit"] = str(
            subprocess.check_output(["git", "rev-parse", "HEAD"])
        )

        args_path = "{0}/input.json".format(kwargs["save_dir"])
        with open(args_path, "w") as f:
            json.dump(kwargs, f, indent=4, sort_keys=True)

        # pop off all of the info that shouldn't go into the main function
        if "git_commit" in kwargs:
            kwargs.pop("git_commit")

        kwargs.pop("kwargs_path")

        # load whatever function you want to run
        main_function = importlib.import_module(kwargs["main_function"])
        kwargs.pop("main_function")

        print(kwargs)

        # run it
        main_function.run(**kwargs)
