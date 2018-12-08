import pytest
import scanpy.api as sc
import geneselection.datasets.dataset as gds


class TestData(object):
    def __init__(self):
        self.dstruct = sc.read('test_data/gene5RowTestData.h5ad')
        #row_two is the first 50 values in row 2
        self.row_two = [6.0664206, 0.81761986, 2.221718, 0.81761986, 6.0939155, 0.0, 0.49014387, 1.2613558,
                        6.1151004, 1.2613558, 1.6915445, 0.0, 1.0639012, 0.49014387, 1.9913629, 0.49014387,
                        4.1726575, 0.81761986, 0.0, 0.49014387, 0.81761986, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.46414,
                        0.49014387, 0.81761986, 1.2613558, 2.7024426, 2.4088187, 0.0, 2.9291346, 1.8017772, 2.9623814,
                        0.0, 3.755331, 1.4261773, 1.2613558, 0.49014387, 0.81761986, 0.0, 1.6915445, 0.49014387,
                        0.81761986, 1.6915445, 1.4261773, 3.9914677]
        self.len_row_two = 50

    def get_annData(self):
        return self.dstruct


@pytest.fixture
def rna_seq():
    return TestData()


def test_len(rna_seq):
    ad = rna_seq.get_annData()
    ds = gds.gsdataset_from_anndata(ad)
    assert len(ds) == 5


def test_access(rna_seq):
    ad = rna_seq.get_annData()
    ds = gds.gsdataset_from_anndata(ad)
    kval = ds[2]
    row = kval['X']
    assert len(row) == 38270


def test_row(rna_seq):
    ad = rna_seq.get_annData()
    ds = gds.gsdataset_from_anndata(ad)
    kval = ds[2]
    row = kval['X']
    for i in range(rna_seq.len_row_two):
        assert row[i] == rna_seq.row_two[i]


