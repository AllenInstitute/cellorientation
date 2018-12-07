import os
import json
import numpy as np
from sklearn.model_selection import train_test_split


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
