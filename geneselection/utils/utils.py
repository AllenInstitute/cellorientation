import argparse
import torch


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# def save_load_dict(save_path, args=None, overwrite=False, verbose=True):
#     # saves a dictionary, 'args', as a json file. Or loads if it exists.

#     if os.path.exists(save_path) and not overwrite:
#         warnings.warn(
#             "args file exists and overwrite is not set to True. Using existing args file."
#         )

#         # load argsions file
#         with open(save_path, "rb") as f:
#             args = json.load(f)
#     else:
#         # make a copy if the args file exists
#         if os.path.exists(save_path):
#             the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
#             shutil.copyfile(save_path, "{0}_{1}".format(save_path, the_time))

#         with open(save_path, "w") as f:
#             json.dump(args, f, indent=4, sort_keys=True)

#     return args


def get_activation(activation):
    if activation is None or activation.lower() == "none":
        return torch.nn.Sequential()

    elif activation.lower() == "relu":
        return torch.nn.ReLU(inplace=True)

    elif activation.lower() == "prelu":
        return torch.nn.PReLU()

    elif activation.lower() == "sigmoid":
        return torch.nn.Sigmoid()

    elif activation.lower() == "leakyrelu":
        return torch.nn.LeakyReLU(0.2, inplace=True)
