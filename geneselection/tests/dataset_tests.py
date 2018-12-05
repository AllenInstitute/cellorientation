import pytest
import scanpy.api.datasets as spd
import geneselection.datasets.dataset as gds


class TestData(object):
    def __init__(self):
        self.dstruct = spd.krumsiek11()

    def get_annData(self):
        return self.dstruct



@pytest.fixture
def cell_fates():
    return TestData()

@pytest.fixture
def rna_seq_without_lablels():
    return gds.get_aics_rnaseq(with_obj_label=False)

@pytest.fixture
def rna_seq_with_lables():
    return gds.get_aics_rnaseq()


def test_len(cell_fates):
    cf = cell_fates.get_annData()
    ds = gds.GSDataset(cf)
    assert len(ds) == 640


def test_access(cell_fates):
    ds = gds.GSDataset(cell_fates.get_annData(), as_tensor=False, with_obj_label=False)
    row = ds[2]
    assert len(row) == 11


def test_rna(rna_seq_without_lablels):
    ds = rna_seq_without_lablels
    data = ds[2]
    #print("keys: ", data[0])
    print("vals: ", data)
    print(data.shape)
    assert len(data) == 38270


def test_rna_with_labels(rna_seq_with_lables):
    ds = rna_seq_with_lables
    objs, vals = ds[2]
    print("objs: ", objs)
    print("len: ", len(objs))
    assert True