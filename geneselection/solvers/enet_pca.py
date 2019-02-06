import multiprocessing
from functools import partial

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

import glmnet_python  # noqa: F401
from glmnet import glmnet

from geneselection.utils.data import tidy

import altair as alt

alt.data_transformers.enable("default", max_rows=None)


def preprocess_cardio(
    adata_in,
    nz_thresh=0.05,
    transform=np.arcsinh,
    f_coding_genes="~/all_human_protein_genes_and_exons.txt",
):

    # load list of protein coding genes
    df = pd.read_csv(f_coding_genes, delimiter="\t")
    coding_genes = [str(g) + "_HUMAN" for g in df["Gene name"].unique()]

    # filter our data for only protein coding genes
    cols = np.array([c for c in adata_in.var.index if c in coding_genes])
    adata = adata_in[:, cols]

    # filter for genes that apoear in at least x% of cells
    gene_nz_freq = (adata.X > 0).mean(axis=0)
    adata = adata[:, cols[gene_nz_freq > nz_thresh]]
    adata.X = transform(adata.X)

    return adata.copy()


def subset_cardio(adata, days=["D0"]):
    if days == "all":
        return adata
    else:
        return adata[adata.obs["day"].isin(days)].copy()


def get_gene_set(df, num_genes=25):
    gene_subset_sizes = df["Number of genes"].unique()
    closest_gene_subset_size = gene_subset_sizes[
        np.abs(gene_subset_sizes - num_genes).argmin()
    ]
    closest_gene_subset = df[df["Number of genes"] == closest_gene_subset_size][
        "Gene name"
    ].values
    return closest_gene_subset


def get_selected_betas(m_fit):
    betas = m_fit["beta"]
    betas_bool = [b != 0 for b in betas]
    for i, a in enumerate(betas_bool):
        for j, b in enumerate(betas_bool):
            if i > j:
                assert np.array_equal(a, b)
    return betas_bool[0]


def enet_pca(
    X,
    n_components=10,
    pc_weights="scaled",
    alpha=0.9,
    penalty_factor=None,
    lambda_path=None,
    **kwargs
):

    # pca and tranform X into top n_components pc coords
    pca = PCA(n_components=n_components, svd_solver="randomized")
    pca.fit(X)
    X_pca = pca.transform(X)

    # weight the pcs or not
    if pc_weights == "variance_explained":
        pass
    elif pc_weights == "scaled":
        X_pca = scale(X_pca)
    else:
        X_pca = pc_weights * scale(X_pca)

    # defalut enet kwargs
    enet_kwargs = {
        "x": X.copy(),
        "y": X_pca.copy(),
        "alpha": alpha,
        "family": "mgaussian",
    }

    # maybe add unevenly weighted penalties
    if penalty_factor is not None:
        enet_kwargs["penalty_factor"] = penalty_factor

    # maybe specify lambda path
    if lambda_path is not None:
        enet_kwargs["lambdau"] = lambda_path

    # run the regression over the lambda path
    mfit = glmnet(**enet_kwargs)

    # return lambda path and which features are selected at those lambdas
    return {"beta": get_selected_betas(mfit), "lambda_path": mfit["lambdau"]}


def filter_out_unpenalized_genes(beta, unpenalized_genes, all_genes):
    beta_out = beta.copy()
    assert beta_out.shape[0] == len(all_genes)
    unpenalized_genes_bool_inds = np.array(
        [gene in set(unpenalized_genes) for gene in all_genes]
    )
    beta_out[unpenalized_genes_bool_inds, :] = False
    return beta_out


def get_selected_genes(
    boot_results,
    adata,
    lambda_index=75,
    selection_threshold_index=75,
    thresholds=np.linspace(0.01, 1, num=10),
    unpenalized_genes=np.array([]),
):
    boot_betas = [
        filter_out_unpenalized_genes(
            beta=br["beta"],
            unpenalized_genes=unpenalized_genes,
            all_genes=[g.split("_")[0] for g in adata.var.index.values],
        )
        for br in boot_results
    ]
    gsel_bool = np.stack(boot_betas).mean(axis=0)
    genes_bool = gsel_bool[:, lambda_index] > thresholds[selection_threshold_index]
    return np.array([g.split("_")[0] for g in adata.var.index.values])[genes_bool]


def thresh_lambda_plot(
    boot_results,
    adata,
    thresholds=np.linspace(0.01, 1, num=10),
    lambdas=np.geomspace(10, 0.01, num=10),
    unpenalized_genes=np.array([]),
):

    boot_betas = [
        filter_out_unpenalized_genes(
            beta=br["beta"],
            unpenalized_genes=unpenalized_genes,
            all_genes=[g.split("_")[0] for g in adata.var.index.values],
        )
        for br in boot_results
    ]
    gsel_bool = np.stack(boot_betas).mean(axis=0)
    gsel_thresh = [
        [(np.sum(gsel_bool[:, j] >= thresh)) for thresh in thresholds]
        for j, a in enumerate(lambdas)
    ]

    df_sel_thresh_alpha = tidy(np.stack(gsel_thresh))
    df_sel_thresh_alpha.columns = [
        "lambda index",
        "selection threshold index",
        "number of genes selected",
    ]
    df_sel_thresh_alpha["log number of genes selected"] = np.log1p(
        df_sel_thresh_alpha["number of genes selected"]
    )

    mychart = (
        alt.Chart(df_sel_thresh_alpha, width=600, height=600)
        .mark_rect()
        .encode(
            alt.X("selection threshold index:O", scale=alt.Scale(paddingInner=0)),
            alt.Y("lambda index:O", scale=alt.Scale(paddingInner=0)),
            alt.Color(
                "log number of genes selected:Q", scale=alt.Scale(scheme="greys")
            ),
        )
    )

    return mychart


def worker(
    boot_inds,
    adata,
    noise=0.001,
    n_pcs=5,
    alpha=0.9,
    lambda_path=np.geomspace(10, 0.01, num=10),
    pc_weights="scaled",
    unpenalized_genes=np.array([]),
):
    X_boot = adata.X[boot_inds, :]
    X_boot = scale(
        scale(X_boot + np.random.normal(scale=noise * 1e-6, size=X_boot.shape))
        + np.random.normal(scale=noise, size=X_boot.shape)
    )
    enet_kwargs = dict(
        n_components=n_pcs,
        alpha=alpha,
        lambda_path=lambda_path,
        pc_weights=pc_weights,
        penalty_factor=np.array(
            [
                gene not in set(unpenalized_genes)
                for gene in [g.split("_")[0] for g in adata.var.index.values]
            ]
        ).astype(float),
    )
    return enet_pca(X_boot, **enet_kwargs)


def parallel_runs(
    adata,
    n_processes=10,
    n_bootstraps=1000,
    noise=0.001,
    n_pcs=5,
    alpha=0.9,
    lambda_path=np.geomspace(10, 0.01, num=10),
    pc_weights="scaled",
    unpenalized_genes=np.array([]),
):

    boot_inds = [
        np.random.choice(len(adata.X), size=len(adata.X)) for _ in range(n_bootstraps)
    ]

    worker_partial = partial(
        worker,
        adata=adata,
        noise=noise,
        n_pcs=n_pcs,
        alpha=alpha,
        lambda_path=lambda_path,
        pc_weights=pc_weights,
        unpenalized_genes=unpenalized_genes,
    )

    pool = multiprocessing.Pool(processes=n_processes)
    result_list = pool.map(worker_partial, boot_inds)
    pool.close()
    pool.join()

    return result_list
