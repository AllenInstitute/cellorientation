import numpy as np
import pandas as pd
import pathlib

from geneselection.utils.data import tidy

import altair as alt

alt.data_transformers.enable("default", max_rows=None)


def preprocess_cardio(
    adata_in,
    nz_thresh=0.05,
    transform=np.arcsinh,
    f_coding_genes=pathlib.PurePath(
        pathlib.Path(__file__).parent.resolve(), "Ensembl_protein_coding_genes.csv"
    ),
    suffix="_HUMAN",
):

    # load list of protein coding genes
    df = pd.read_csv(f_coding_genes)
    coding_genes = [str(g) + suffix for g in df["Gene name"].unique()]

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


def hub_persistence_plot(
    adata, boot_results, downsample_backgound=1000, width=400, height=400
):

    z = np.stack([br["beta"] for br in boot_results])
    sel_fracs = (z != 0).mean(axis=0)
    df_sel = tidy(sel_fracs)
    df_sel.columns = ["Gene Index", "Lambda Index", "Selection Fraction"]

    df_tmp = adata.var.copy()
    df_tmp["Gene Index"] = df_tmp.index.values.astype(np.int64)
    df_sel = df_sel.merge(df_tmp)
    df_sel = df_sel[df_sel["Gene Index"] < 1000 + downsample_backgound]

    chart = (
        alt.Chart(df_sel, width=width, height=height)
        .mark_line(opacity=0.25, interpolate="linear", color="blue")
        .encode(
            x="Lambda Index:Q",
            y="Selection Fraction:Q",
            detail="Gene Index:N",
            color="Type:N",
        )
    )

    return chart
