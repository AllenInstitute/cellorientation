import numpy as np
import pandas as pd
import scanpy.api as sc
from scipy.io import mmread
from tqdm import tqdm

import requests
import os


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


def load(
    loc="data_files",
    blocksize=1000000,
    anndata_write=True,
    anndata_name="mouse_retina.h5ad",
    X_dtype=np.float32,
):

    adata_fpath = os.path.join(loc, anndata_name)

    # if we've already down;loaded and constructed the adata file, read it and use it
    if os.path.exists(adata_fpath) and os.path.isfile(adata_fpath):
        print("reading saved anndata h5ad file")
        adata = sc.read_h5ad(adata_fpath)

    # if anndata doesn't exit alread, download inputs and construct it
    else:
        # download files if they don't exist locally
        if not os.path.exists(loc):
            os.makedirs(loc)
        files = {
            "10x_mouse_retina_development.mtx": "https://www.dropbox.com/s/6d76z4grcnaxgcg/10x_mouse_retina_development.mtx?dl=1",
            "10x_mouse_retina_development_phenotype.csv": "https://www.dropbox.com/s/y5lho9ifzoktjcs/10x_mouse_retina_development_phenotype.csv?dl=1",
            "10x_mouse_retina_development_feature.csv": "https://www.dropbox.com/s/1mc4geu3hixrxhj/10x_mouse_retina_development_feature.csv?dl=1",
        }
        print("downloading data files")
        for fname, url in files.items():
            if not os.path.exists(os.path.join(loc, fname)):
                download_file(url, loc=loc, blocksize=blocksize)

        # read in data
        print("reading data files")
        df_obs = pd.read_csv(
            os.path.join(loc, "10x_mouse_retina_development_phenotype.csv"), index_col=0
        )[["barcode", "sample", "age", "CellType"]]
        df_var = pd.read_csv(
            os.path.join(loc, "10x_mouse_retina_development_feature.csv"), index_col=0
        )[["id", "gene_short_name"]]
        count_mat = mmread(os.path.join(loc, "10x_mouse_retina_development.mtx"))

        # make anndata object
        print("constructing anndata object")
        adata = sc.AnnData(
            X=count_mat.toarray().astype(X_dtype).transpose(), obs=df_obs, var=df_var
        )
        genes_to_keep = np.mean(adata.X != 0, axis=0) > 0
        cells_to_keep = np.mean(adata.X != 0, axis=1) > 0
        adata = adata[:, genes_to_keep][cells_to_keep, :].copy()

        # save a local copy
        if anndata_write:
            print("saving annndata h5ad file")
            adata.write(adata_fpath)

    return adata
