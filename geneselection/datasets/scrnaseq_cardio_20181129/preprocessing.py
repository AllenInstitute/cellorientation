import numpy as np
import scanpy.api as sc
import os
import json
from sklearn.model_selection import train_test_split


def load_original(
    loc="/allen/aics/modeling/data/scRNAseq_SeeligCollaboration/data_for_modeling/scrnaseq_cardio_20181129.h5ad"
):
    return sc.read_h5ad(loc)


def make_splits(
    out_file="geneselection/datasets/scrnaseq_cardio_20181129/scrnaseq_cardio_20181129_80_20_split.json"
):
    adata = load_original()
    all_inds = np.arange(len(adata))
    inds_train, inds_test = train_test_split(all_inds, test_size=0.2, random_state=0)
    split_dict = {"train": sorted(inds_train), "test": sorted(inds_test)}
    split_dict = {k: [int(i) for i in v] for k, v in split_dict.items()}
    with open(out_file, "w") as f:
        json.dump(split_dict, f)


def split_original(
    in_loc="/allen/aics/modeling/data/scRNAseq_SeeligCollaboration/data_for_modeling/scrnaseq_cardio_20181129.h5ad",
    out_loc="/allen/aics/modeling/data/scRNAseq_SeeligCollaboration/data_for_modeling/",
    split_loc="geneselection/datasets/scrnaseq_cardio_20181129/scrnaseq_cardio_20181129_80_20_split.json",
):
    adata = load_original(in_loc)

    with open(split_loc, "r") as f:
        d = json.load(f)

    adata_train, adata_test = adata[d["train"], :].copy(), adata[d["test"], :].copy()
    adata_train.write(os.path.join(out_loc, "scrnaseq_cardio_20181129_train.h5ad"))
    adata_test.write(os.path.join(out_loc, "scrnaseq_cardio_20181129_test.h5ad"))
