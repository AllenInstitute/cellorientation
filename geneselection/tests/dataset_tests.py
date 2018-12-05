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


def test_len(cell_fates):
    cf = cell_fates.get_annData()
    ds = gds.GSDataset(cf)
    assert len(ds) == 640


def test_access(cell_fates):
    ds = gds.GSDataset(cell_fates.get_annData(), as_tensor=False, with_obj_label=False)
    row = ds[2]
    assert len(row) == 11
