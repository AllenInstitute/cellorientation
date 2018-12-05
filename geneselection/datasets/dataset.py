import anndata
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch import Tensor
from collections import Iterable
from ..utils.dataloader import default_collate
from typing import Dict, List, Union





class GSDataset(Dataset):
    def __init__(
        self,
        x: Tensor=torch.zeros(1, 1),
        obs: pd.DataFrame=pd.DataFrame([0]),
        var: pd.DataFrame=pd.DataFrame([0]),
        uns: dict={}
    ):
        """
        A data provider class for the larger project. The idea is to capture an AnnData and then
        serve up the rows as either pytorch Tensors, or numpy ndarrays based on the keyword argument
        :param X: a Tensor created from for example an (ndarray)AnnData.X data block
        :param obs: a pandas.DataFrame containing the observational labels corresponding to AnnData.obs
        :param var: a pandas.DataFrame containing the variable labels contained in an AnnData.obs
        :param uns: a pandas.DataFrame containing the unstructured descriptors contained in AnnData.uns
        """
        super(GSDataset, self).__init__()

        N, D = x.shape
        assert N == len(obs) and D == len(var)

        self.X = x
        self.obs = obs
        self.var = var
        self.uns = uns

    def __len__(self) -> int:
        return len(self.X)

    def _get_item(self, idx: int) -> Dict[Tensor, pd.Series]:
        """
        Helper function to return a dictionary of {one row of X, obs for row} for one index
        :param idx: index of row to return
        :return: dict(Tensor, DataFrame)
        """
        X = self.X[idx]
        obs = self.obs.iloc[[idx]]
        return dict(X=X, obs=obs)

    def __getitem__(self, idx: Union[int, List]) -> Union[Dict[Tensor, pd.Series], List[Dict[Tensor, pd.Series]]]:
        return (
            default_collate([self._get_item(i) for i in idx])
            if (isinstance(idx, Iterable) and not isinstance(idx, str))
            else self._get_item(idx)
        )

    def __add__(self, other: 'GSDataset'):
        assert self.var.equals(other.var)
        return GSDataset(
            X=torch.cat([self.X, other.X]),
            obs=pd.concat([self.obs, other.obs]),
            var=self.var,
            uns={**self.uns, **other.uns},
        )


def gsdataset_from_anndata(adata: anndata.AnnData) -> GSDataset:
    return GSDataset(
        X=torch.from_numpy(adata.X), obs=adata.obs, var=adata.var, uns=adata.uns
    )
