import string
import random
import pandas as pd
import numpy as np
import anndata


class CreateH5ad(anndata.AnnData):
    """
    Create mock data for H5ad data format
    """
    def __new__(cls, rows: int = 5, columns: int = 38270):
        return anndata.AnnData(X=CreateH5ad.create_x_data(rows, columns),
                               obs=CreateH5ad.create_obs(rows),
                               var=CreateH5ad.create_var(columns)
                               )

    @staticmethod
    def create_x_data(rows: int, columns: int):
        return np.random.rand(rows, columns)

    @staticmethod
    def create_var(columns):
        s = set()
        while len(s) < columns:
            s.add(CreateH5ad.create_random_id())
        data = {'gene.mean': np.random.rand(1, columns)[0], 'gene.dispersion': np.random.rand(1, columns)[0] }
        df = pd.DataFrame(data=data, index=s)
        return df

    @staticmethod
    def create_random_id(prefix_len: list = [3, 4, 5, 6], common_suffix: str = '_HUMAN') -> str:
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(random.choice(prefix_len))) + common_suffix

    @staticmethod
    def create_obs(rows):
        return CreateH5ad.create_random_obs(rows)

    @staticmethod
    def create_random_obs(rows: int, prefix: str = "E1_") -> str:
        ans = [prefix.join(x).join('_' + '_'.join(CreateH5ad.random_ints())) for x in CreateH5ad.random_dna_set(rows)]
        return ans

    @staticmethod
    def random_ints(length: int = 2) -> set:
        s = set()
        while len(s) < length:
            s.add(str(random.choice(range(100))))
        return s

    @staticmethod
    def random_dna_set(set_size: int, length: int = 16, bases=['A', 'C', 'T', 'G']):
        s = set()
        while len(s) < set_size:
            s.add(''.join(random.choice(bases) for _ in range(length)))
        return s
