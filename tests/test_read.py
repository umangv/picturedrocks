import numpy as np
import picturedrocks as pr
import scanpy.api as sc

def test_process_clusts():
    adata = sc.datasets.paul15()
    pr.read.process_clusts(adata, "paul15_clusters")
    assert adata.obs['clust'].dtype.name == "category"
    assert np.issubdtype(adata.obs['y'].dtype, np.integer)
    assert len(adata.obs['y'].unique()) == len(adata.obs['clust'].cat.categories)
