import numpy as np
import pandas as pd
import scanpy.api as sc
import os
from scipy.io import mmread
from ..utils.data import download_file, split_anndata, write_splits


def _create_anndata(
    anndata_name_out="mouse_retina.h5ad",
    X_dtype=np.float32,
    cache_dir="data_files",
    blocksize=1000000,
):

    adata_fpath = os.path.join(cache_dir, anndata_name_out)

    # if we've already downloaded and constructed the adata file, read it and use it
    if os.path.exists(adata_fpath) and os.path.isfile(adata_fpath):
        print("reading saved anndata h5ad file")
        adata = sc.read_h5ad(adata_fpath)

    # if anndata doesn't exist already, download inputs and construct it
    else:
        # download files if they don't exist locally
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        files = {
            "10x_mouse_retina_development.mtx": "https://www.dropbox.com/s/6d76z4grcnaxgcg/10x_mouse_retina_development.mtx?dl=1",
            "10x_mouse_retina_development_phenotype.csv": "https://www.dropbox.com/s/y5lho9ifzoktjcs/10x_mouse_retina_development_phenotype.csv?dl=1",
            "10x_mouse_retina_development_feature.csv": "https://www.dropbox.com/s/1mc4geu3hixrxhj/10x_mouse_retina_development_feature.csv?dl=1",
        }
        print("downloading data files")
        for fname, url in files.items():
            if not os.path.exists(os.path.join(cache_dir, fname)):
                download_file(url, loc=cache_dir, blocksize=blocksize)

        # read in data
        print("reading data files")
        df_obs = pd.read_csv(
            os.path.join(cache_dir, "10x_mouse_retina_development_phenotype.csv"),
            index_col=0,
        )[["barcode", "sample", "age", "CellType"]]
        df_var = pd.read_csv(
            os.path.join(cache_dir, "10x_mouse_retina_development_feature.csv"),
            index_col=0,
        )[["id", "gene_short_name"]]
        count_mat = mmread(os.path.join(cache_dir, "10x_mouse_retina_development.mtx"))

        # make anndata object
        print("constructing anndata object")
        adata = sc.AnnData(
            X=count_mat.toarray().astype(X_dtype).transpose(), obs=df_obs, var=df_var
        )

        print("removing zero-count cells and genes")
        genes_to_keep = np.mean(adata.X != 0, axis=0) > 0
        cells_to_keep = np.mean(adata.X != 0, axis=1) > 0
        adata = adata[:, genes_to_keep][cells_to_keep, :].copy()

        # save a local copy
        print("saving annndata h5ad file")
        adata.write(adata_fpath)


def load(
    anndata_name_original="mouse_retina.h5ad",
    split="train",
    cache_dir="data_cache",
    cache=True,
):
    """
    Load requested split of mouse data, where the whole dataset.
    Looks for a local cache of the original data, and creates it in cache_dir if not there and cache=True.
    Then Looks for local cache of the requested split, and if it can't find that, makes a split on the fly.
    If cache=True, caches the result in cache_dir for next time."""

    original_fpath = os.path.join(cache_dir, anndata_name_original)
    if not os.path.exists(original_fpath):
        _create_anndata(anndata_name_out=anndata_name_original, cache_dir=cache_dir)

    original_fname = os.path.basename(original_fpath)
    original_bname, original_ext = os.path.splitext(original_fname)
    target_fname = "{0}_{1}{2}".format(original_bname, split, original_ext)
    target_fpath = os.path.join(cache_dir, target_fname)

    if not os.path.exists(target_fpath):
        adata_in = sc.read_h5ad(original_fpath)
        split_inds, split_adata = split_anndata(adata_in)
        if cache:
            write_splits(
                split_inds_dict=split_inds,
                split_adata_dict=split_adata,
                basename=original_bname,
                out_dir=cache_dir,
            )

    return sc.read_h5ad(target_fpath)
