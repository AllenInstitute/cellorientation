import scanpy.api as sc
from .dataset import GSDataset


def load(
    path="/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/data_for_modeling/scrnaseq_cardio_20181129.h5ad",
    **kwargs
):
    adata = sc.read(path)

    return GSDataset(adata, **kwargs)
