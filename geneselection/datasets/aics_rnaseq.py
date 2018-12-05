import scanpy.api as sc

def load(path = '/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/data_for_modeling/scrnaseq_cardio_20181129.h5ad'):
    adata = sc.read(path)
    
    return path
    