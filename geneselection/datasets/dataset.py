import anndata
import pandas as pd
from collections import Iterable
from ..utils.dataloader import default_collate
from typing import Dict, List, Union, Mapping, Any
from torch import Tensor, from_numpy, cat
from torch.utils.data.dataset import Dataset


class GSDatasetVarMismatchError(Exception):
    """Exception for when asked to merge 2 GSDatasets with different var classes (columns)"""

    pass


class GSDataset(Dataset):
    def __init__(
        self,
        X: Tensor = Tensor([[0]]),
        obs: pd.DataFrame = pd.DataFrame([0]),
        var: pd.DataFrame = pd.DataFrame([0]),
        uns: Mapping[Any, Any] = {},
    ):
        """
        A data provider class for the larger project. The idea is to capture an AnnData and then
        serve up the rows as either pytorch Tensors, or numpy ndarrays based on the keyword argument
        :param X: a Tensor created from for example an (ndarray)AnnData.X data block
        :param obs: a pandas.DataFrame containing the observational labels corresponding to AnnData.obs
        :param var: a pandas.DataFrame containing the variable labels contained in an AnnData.obs
        :param uns: a dictionary or DataFrame or other Mapping containing the unstructured descriptors (AnnData.uns)
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

    def _get_item(self, idx: int) -> Dict[Tensor, pd.DataFrame]:
        """
        Helper function to return a dictionary of {one row of X, obs for row} for one index
        :param idx: index of row to return
        :return: dict(Tensor, DataFrame)
        """
        X = self.X[idx]
        obs = self.obs.iloc[[idx]]
        return dict(X=X, obs=obs, idx=idx)

    def __getitem__(
        self, idx: Union[int, List]
    ) -> Union[Dict[Tensor, pd.DataFrame], List[Dict[Tensor, pd.Series]]]:
        return (
            default_collate([self._get_item(i) for i in idx])
            if (isinstance(idx, Iterable) and not isinstance(idx, str))
            else self._get_item(idx)
        )

    def __add__(self, other: "GSDataset"):
        if not self.var.equals(other.var):
            raise GSDatasetVarMismatchError
        return GSDataset(
            X=cat([self.X, other.X]),
            obs=pd.concat([self.obs, other.obs]),
            var=self.var,
            uns={**self.uns, **other.uns},
        )


def gsdataset_from_anndata(adata: anndata.AnnData) -> GSDataset:
    return GSDataset(X=from_numpy(adata.X), obs=adata.obs, var=adata.var, uns=adata.uns)
