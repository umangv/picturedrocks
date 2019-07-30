import numpy as np
import pandas as pd
import picturedrocks as pr
try:
    import scanpy as sc
    sc.pp
except AttributeError:
    sc = sc.api
import anndata
import pytest


def _get_version_info(v):
    def try_to_int(x):
        try:
            return int(x)
        except:
            return x
    return tuple([try_to_int(x) for x in v.split(".")])


@pytest.fixture()
def mock_dataset(datadir):
    adata = sc.read_csv(str(datadir / "test-dataset.csv"))
    return adata


def test_process_clusts():
    adata = sc.datasets.paul15()
    pr.read.process_clusts(adata, "paul15_clusters")
    assert adata.obs["clust"].dtype.name == "category"
    assert np.issubdtype(adata.obs["y"].dtype, np.integer)
    assert len(adata.obs["y"].unique()) == len(adata.obs["clust"].cat.categories)


def test_read_clusts_w_wo_head(mock_dataset, datadir):
    adata_raw = mock_dataset.copy()
    pr.read.read_clusts(adata_raw, str(datadir / "test-labels-raw.csv"), header=False)
    adata_raw_head = mock_dataset.copy()
    pr.read.read_clusts(adata_raw_head, str(datadir / "test-labels-raw-head.csv"))
    assert not adata_raw.obs["clust"].isnull().any()
    assert not adata_raw_head.obs["clust"].isnull().any()
    assert adata_raw.obs["clust"].equals(adata_raw_head.obs["clust"])
    assert np.array_equal(adata_raw.obs["clust"].values, [f"Cluster {i}" for i in np.arange(4) % 2])


def test_read_clusts_raw_and_meta(mock_dataset, datadir):
    adata_raw = mock_dataset.copy()
    pr.read.read_clusts(adata_raw, str(datadir / "test-labels-raw.csv"), header=False)
    adata_meta = mock_dataset.copy()
    pr.read.read_clusts(adata_meta, str(datadir / "test-labels-meta.csv"))
    assert adata_raw.obs["clust"].equals(adata_meta.obs["clust"])


@pytest.mark.skipif(
    _get_version_info(anndata.__version__) < (0, 6, 16), reason="Need new adata"
)
def test_read_clusts_with_shuffled_rows(mock_dataset, datadir):
    adata_orig = mock_dataset.copy()
    pr.read.read_clusts(adata_orig, str(datadir / "test-labels-raw.csv"), header=False)
    df = pd.read_csv(datadir / "test-dataset.csv", index_col=0)
    shuffle = [3, 0, 2, 1]
    adata = anndata.AnnData(df.iloc[shuffle].copy())
    pr.read.read_clusts(adata, str(datadir / "test-labels-meta.csv"))
    assert adata.obs["clust"].sort_index().equals(adata_orig.obs["clust"])
    adata2 = anndata.AnnData(df.iloc[shuffle].copy())
    pr.read.read_clusts(adata2, str(datadir / "test-labels-raw.csv"), header=False)
    assert not adata2.obs["clust"].equals(adata.obs["clust"].sort_index())

