from torch.utils.data.dataset import Dataset
import anndata
from numpy import ndarray
from torch import Tensor

import scanpy.api as sc

class GSDataset(Dataset):
    def __init__(self, data: anndata.AnnData, as_tensor: bool=True, with_obj_label: bool=True):
        """
        A data provider class for the larger project. The idea is to capture an AnnData and then
        serve up the rows as either pytorch Tensors, or numpy ndarrays based on the keyword argument
        :param data: An AnnData object which is copied to protect from changes to the original object.
        :param as_tensor: If True the access method returns rows as Tensors if False it returns ndarrays
        :param with_obj_label: If true return Tuple(obj_label: str, Tensor or ndarray)
        """
        super(GSDataset, self).__init__()
        self.data = data.copy()
        self.return_tensor=as_tensor
        self.with_obj_label=with_obj_label

    def __len__(self) -> int:
        return self.data.n_obs

    def __getitem__(self, index: int) -> [ndarray, Tensor, (ndarray, ndarray), (ndarray, Tensor)]:
        row = None
        if not self.return_tensor:
            row = self.data.X[index]
        else:
            row = Tensor(self.data.X[index])
        if not self.with_obj_label:
            return row
        keys = self.data.obs.iloc[index, :]
        return (keys, row)

    def __add__(self, other):
        self.data.concatenate(other)

def get_aics_rnaseq(path = '/allen/aics/gene-editing/RNA_seq/scRNAseq_SeeligCollaboration/data_for_modeling/scrnaseq_cardio_20181129.h5ad', **kwargs):
    adata = sc.read(path)
    
    return GSDataset(adata, **kwargs)