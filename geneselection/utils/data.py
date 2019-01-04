import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing


def download_file(url, loc="data_files", blocksize=1000000):
    if not os.path.exists(loc):
        os.makedirs(loc)
    local_filename = os.path.join(loc, url.split("/")[-1])
    if "?dl" in local_filename:
        local_filename = local_filename.split("?dl")[0]
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=blocksize), unit="MB"):
                if chunk:
                    f.write(chunk)


def split_anndata(adata, test_size=0.2, random_state=0):
    """Split anndata into train/test and return a dict of the integer indices (from the original anndata) in each split and a dict of each split anndata"""
    all_inds = np.arange(len(adata))
    inds_train, inds_test = train_test_split(
        all_inds, test_size=test_size, random_state=random_state
    )
    split_inds_dict = {"train": sorted(inds_train), "test": sorted(inds_test)}
    split_inds_dict = {k: [int(i) for i in v] for k, v in split_inds_dict.items()}
    return split_inds_dict, {k: adata[v, :] for k, v in split_inds_dict.items()}


def write_splits(
    split_inds_dict=None, split_adata_dict=None, out_dir="./", basename="dataset_foo"
):
    """save anndata splits to disk in out_dir"""

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if split_inds_dict is not None:
        inds_filename = "{0}_split_inds.json".format(basename)
        with open(os.path.join(out_dir, inds_filename), "w") as f:
            json.dump(split_inds_dict, f)

    if split_adata_dict is not None:
        for split, adata in split_adata_dict.items():
            adata_filename = "{0}_{1}.h5ad".format(basename, split)
            adata.write(os.path.join(out_dir, adata_filename))


def tidy(arr):
    """Take a numpy ndarray and turn it into a tidy dataframe."""
    return pd.DataFrame([(*inds, arr[inds]) for inds in np.ndindex(arr.shape)])


def transform(ds_train, ds_validate, method=None):
    """Method can be None, 'none', 'log', 'zscore', 'box-cox'"""
    if method is None or method == "none":
        pass  # do nothing

    elif method == "log":
        ds_train.X = np.log(ds_train.X + 1)
        ds_validate.X = np.log(ds_validate.X + 1)

    elif method == "box-cox":
        ds_train.X = preprocessing.power_transform(ds_train.X + 1)
        ds_validate.X = preprocessing.power_transform(ds_validate.X + 1)

    elif method == "zscore":
        X = ds_train.X

        mu = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        ds_train.X = (X - mu) / std
        ds_validate.X = (ds_validate.X - mu) / std

    else:
        raise ValueError("method value of {} is not supported".format(method))

    return ds_train, ds_validate
