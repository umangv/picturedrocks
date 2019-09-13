import numpy as np
import picturedrocks as pr
try:
    import scanpy as sc
    sc.pp
except AttributeError:
    sc = sc.api
import pytest
import importlib

ipyw_spec = importlib.util.find_spec("ipywidgets")

@pytest.fixture()
def pauldata():
    adata = sc.datasets.paul15()
    pr.read.process_clusts(adata, "paul15_clusters")
    infoset = pr.markers.makeinfoset(adata, True)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return (adata, infoset)


@pytest.mark.skipif(ipyw_spec is None, reason="ipywidgets not installed (optional dependency)")
def test_empty_visuals(pauldata):
    adata, infoset = pauldata
    cife = pr.markers.CIFE(infoset)
    im = pr.markers.interactive.InteractiveMarkerSelection(adata, cife, [])
    im.show()

@pytest.mark.skipif(ipyw_spec is None, reason="ipywidgets not installed (optional dependency)")
def test_tsne_violin(pauldata):
    adata, infoset = pauldata
    cife = pr.markers.CIFE(infoset)
    im = pr.markers.interactive.InteractiveMarkerSelection(adata, cife,["tsne","violin"])
    im.show()

@pytest.mark.skip(reason="current version runs too slowly")
def test_umap_violin(pauldata):
    adata, infoset = pauldata
    cife = pr.markers.CIFE(infoset)
    im = pr.markers.interactive.InteractiveMarkerSelection(adata, cife,["tsne","violin"])
    im.show()