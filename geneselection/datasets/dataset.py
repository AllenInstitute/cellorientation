import anndata
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from collections import Iterable
from ..utils.dataloader import default_collate


def gsdataset_from_anndata(adata: anndata.AnnData):
    return GSDataset(X=adata.X, obs=adata.obs, var=adata.var, uns=adata.uns)


class GSDataset(Dataset):
    def __init__(
        self, X=torch.zeros(1, 1), obs=pd.DataFrame([0]), var=pd.DataFrame([0]), uns={}
    ):
        """
        A data provider class for the larger project. The idea is to capture an AnnData and then
        serve up the rows as either pytorch Tensors, or numpy ndarrays based on the keyword argument
        :param data: An AnnData object which is copied to protect from changes to the original object.
        """
        super(GSDataset, self).__init__()

        N, D = X.shape
        assert N == len(obs) and D == len(var)

        self.X = X
        self.obs = obs
        self.var = var
        self.uns = uns

    def __len__(self) -> int:
        return len(self.X)

    def _get_item(self, idx):
        X = self.X[idx]
        obs = self.obs.iloc[[idx]]
        return dict(X=X, obs=obs)

    def __getitem__(self, idx):
        return (
            default_collate([self._get_item(i) for i in idx])
            if (isinstance(idx, Iterable) and not isinstance(idx, str))
            else self._get_item(idx)
        )

    def __add__(self, other):
        assert self.var.equals(other.var)
        return GSDataset(
            X=torch.cat([self.X, other.X]),
            obs=pd.concat([self.obs, other.obs]),
            var=self.var,
            uns={**self.uns, **other.uns},
        )
