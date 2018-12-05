import anndata
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from collections import Iterable
from ..utils.dataloader import default_collate


class GSDataset(Dataset):
    def __init__(self, data: anndata.AnnData):
        """
        A data provider class for the larger project. The idea is to capture an AnnData and then
        serve up the rows as either pytorch Tensors, or numpy ndarrays based on the keyword argument
        :param data: An AnnData object which is copied to protect from changes to the original object.
        """
        super(GSDataset, self).__init__()
        self.X = torch.from_numpy(data.X.copy())
        self.obs = data.obs.copy()
        self.var = data.var.copy()
        self.uns = data.uns.copy()

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
            anndata.AnnData(
                X=torch.cat([self.X, other.X]).numpy(),
                obs=pd.concat([self.obs, other.obs]),
                var=self.var,
                uns={**self.uns, **other.uns},
            )
        )
