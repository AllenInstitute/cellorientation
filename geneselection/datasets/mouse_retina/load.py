import numpy as np
import pandas as pd
import networkx as nx
import scanpy.api as sc
from natsort import natsorted
from scipy.io import mmread
from scipy.stats import boxcox
from tqdm import tqdm
import warnings

import requests
import os


def download_file(url, loc="data_files", blocksize=1024):
    if not os.path.exists(loc):
        os.makedirs(loc)
    local_filename = os.path.join(loc,url.split('/')[-1])
    if "?dl" in local_filename:
        local_filename = local_filename.split("?dl")[0]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=blocksize)):
                if chunk:
                    f.write(chunk)
    return local_filename


def load(loc="data_files", blocksize=1024, anndata_write=True, anndata_name="mouse_retina"):
    
    adata_fpath = os.path.join(loc, anndata_name)
    
    # if we've already down;loaded and constructed the adata file, read it and use it
    if os.path.exists(adata_fpath) and os.path.isfile(adata_fpath):
        adata = sc.read_h5ad(adata_fpath)

    # if anndata doesn't exit alread, download inputs and construct it
    else:
        # download files if they don't exist locally
        if not os.path.exists(loc):
            os.makedirs(loc)
        files={"10x_mouse_retina_development.mtx":"https://www.dropbox.com/s/6d76z4grcnaxgcg/10x_mouse_retina_development.mtx?dl=1",
               "10x_mouse_retina_development_phenotype.csv":"https://www.dropbox.com/s/y5lho9ifzoktjcs/10x_mouse_retina_development_phenotype.csv?dl=1",
               "10x_mouse_retina_development_feature.csv":"https://www.dropbox.com/s/1mc4geu3hixrxhj/10x_mouse_retina_development_feature.csv?dl=1"}
        for fname, url in files.items():
            if not os.path.exists(os.path.join(loc,fname)):
                _ = download_file(loc=loc, blocksize=blocksize)

        # read in data
        df_obs = pd.read_csv("data/10x_mouse_retina_development_phenotype.csv", index_col=0)[
        ["barcode", "sample", "age", "CellType"]]
        df_var = pd.read_csv("data/10x_mouse_retina_development_feature.csv", index_col=0)[
        ["id", "gene_short_name"]]
        count_mat = mmread("data/10x_mouse_retina_development.mtx")

        # make anndata object
        adata = sc.AnnData(X=count_mat.toarray().transpose(), obs=df_obs, var=df_var)
        genes_to_keep = np.mean(adata.X != 0, axis=0) > 0
        cells_to_keep = np.mean(adata.X != 0, axis=1) > 0
        adata = adata[:, genes_to_keep][cells_to_keep, :].copy()

        # save a local copy
        if write_anndata:
            adata.write(adata_fpath)
    
    return adata
