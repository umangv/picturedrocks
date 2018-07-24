# Copyright © 2017, 2018 Anna Gilbert, Alexander Vargo, Umang Varma
#
# This file is part of PicturedRocks.
#
# PicturedRocks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PicturedRocks is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PicturedRocks.  If not, see <http://www.gnu.org/licenses/>.

import pandas as pd

from anndata import AnnData
from warnings import warn
import numpy as np


def read_clusts(adata, filename, sep=","):
    clustdf = pd.read_csv(filename, sep=sep)
    if clustdf.shape[1] == 2:
        clustdf = clustdf.set_index(clustdf.columns[0])
    assert clustdf.shape[1] == 1, "Cluster column ambigious"
    clustser = clustdf.iloc[:, 0]
    if clustser.dtype.kind == "i":
        if clustser.min() > 0:
            warn("Changing cluster ids to begin at 0.")
            clustser -= clustser.min()
        clustuniq = np.sort(clustser.unique())
        assert np.array_equal(
            clustuniq, np.arange(clustuniq.size)
        ), "Cluster ids need to be 0, 1, ..., K-1"
        adata.obs["y"] = clustser
        adata.obs["clust"] = ("Cluster " + clustser.astype("str")).astype("category")
    else:
        adata.obs["clust"] = clustser.astype("category")
        adata.obs["y"] = adata.obs["clust"].cat.codes
    if adata.obs["y"].isnull().any() or adata.obs["clust"].isnull().any():
        warn("Some or all cells not assigned to cluster.")
    adata.uns["num_clusts"] = adata.obs["clust"].cat.categories.size
    clusterindices = {}
    for k in range(adata.uns["num_clusts"]):
        clusterindices[k] = (adata.obs["y"] == k).nonzero()[0]
    adata.uns["clusterindices"] = clusterindices
