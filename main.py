import json
import os
import importlib
import argparse
import datetime
import subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None, help="gpu id")
    parser.add_argument(
        "--kwargs_path", type=str, default=None, help="kwargs for the classifier"
    )

    args = vars(parser.parse_args())

    if os.path.exists(args["kwargs_path"]):
        with open(args["kwargs_path"], "rb") as f:
            kwargs = json.load(f)

    kwargs["kwargs_path"] = args["kwargs_path"]
    kwargs["gpu_ids"] = args["gpu_ids"]

    if "save_parent" in kwargs:
        the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        kwargs["save_dir"] = os.path.join(kwargs["save_parent"], the_time)
        kwargs.pop("save_parent")

    if not os.path.exists(kwargs["save_dir"]):
        os.makedirs(kwargs["save_dir"])

    kwargs["git_commit"] = str(subprocess.check_output(["git", "rev-parse", "HEAD"]))

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

    # discard gpu_ids variable if not needed
    if kwargs["gpu_ids"] is None:
        kwargs.pop("gpu_ids")

    # run it
    main_function.run(**kwargs)
