import scanpy.api as sc


def log_and_zscore(adata):
    """
    normalizes adata.X in place.
    takes an andata with X as raw counts and appliees the following funcitons to X in succession:
      - log1p all entries
      - scale columns (genes) to zero mean and unit variance
    """

    sc.pp.log1p(adata)
    sc.pp.scale(adata)


def normpercell_and_log_and_zscore(adata):
    """
    normalizes adata.X in place.
    takes an andata with X as raw counts and appliees the following funcitons to X in succession:
      - normalize transcript counts per cell
      - log1p all entries
      - scale columns (genes) to zero mean and unit variance
    """

    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
