import pytest
import geneselection.datasets.dataset as gds
from .create_h5ad import CreateH5ad

class StubData(object):
    def __init__(self):
        self.rows = 5
        self.columns = 38270
        self.dstruct = CreateH5ad(rows=self.rows, columns=self.columns)

    def get_annData(self):
        return self.dstruct


@pytest.fixture
def rna_seq():
    return StubData()


def test_len(rna_seq):
    ad = rna_seq.get_annData()
    ds = gds.gsdataset_from_anndata(ad)
    assert len(ds) == rna_seq.rows


def test_access(rna_seq):
    ad = rna_seq.get_annData()
    ds = gds.gsdataset_from_anndata(ad)
    kval = ds[2]
    row = kval['X']
    assert len(row) == rna_seq.columns


def test_row(rna_seq):
    ad = rna_seq.get_annData()
    ds = gds.gsdataset_from_anndata(ad)
    kval = ds[2]
    row = kval['X']
    for i in range(rna_seq.columns):
        assert float(row[i]) == rna_seq.dstruct.X[2, i]


