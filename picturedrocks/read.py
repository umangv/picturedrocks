# Copyright © 2017-2019 Umang Varma, Anna Gilbert
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


def read_clusts(adata, filename, sep=",", name="clust", header=True, copy=False):
    """Read cluster labels from a csv into an obs column

    Args
    ----
    adata: anndata.AnnData
        the `AnnData` object to read labels into
    filename: str
        filename of the csv file with labels
    sep: str, optional
        csv delimiter 
    name: str, optional
        destination for label is adata.obs[name]
    header: bool
        deterimes whether csv has a header line. If false, it is assumed that
        data begins at the first line of csv
    copy: bool
        determines whether a copy of `AnnData` object is returned

    Returns
    -------
    anndata.AnnData
        object with cluster labels
    
    Notes
    -----
     * Cluster ids will automatically be changed so they are 0-indexed
     * csv can either be two columns (in which case the first column is treated
       as observation label and merging handled by pandas) or one column (only
       cluster labels, ordered as in ``adata``)
    """
    adata = adata.copy() if copy else adata
    header = 0 if header else None
    clustdf = pd.read_csv(filename, sep=sep, header=header)
    if clustdf.shape[1] == 2:
        clustdf = clustdf.set_index(clustdf.columns[0])
    elif clustdf.shape[1] == 1:
        clustdf.index = adata.obs.index
    assert clustdf.shape[1] == 1, "Cluster column ambigious"
    clusters = clustdf.iloc[:, 0]
    if clusters.dtype.kind == "i":
        adata.obs[name] = ("Cluster " + clusters.astype("str")).astype("category")
    else:
        adata.obs[name] = clusters.astype("category")
    if adata.obs[name].isnull().any():
        warn("Some or all cells not assigned to cluster.")
    return adata


def process_clusts(adata, name="clust", copy=False):
    """Process cluster labels from an obs column

    This copies `adata.obs[name]` into `adata.obs["clust"]` and precomputes
    cluster indices, number of clusters, etc for use by various functions in
    PicturedRocks.

    Args
    ----
    adata: anndata.AnnData
    copy: bool
        determines whether a copy of `AnnData` object is returned

    Return
    -------
    anndata.AnnData
        object with annotation

    Notes
    ------
    The information computed here is lost when saving as a `.loom` file. If a
    `.loom` file has cluster information, you should run this function
    immediately after :func:`sc.read_loom <scanpy.api.read_loom>`.
    """
    adata = adata.copy() if copy else adata
    adata.obs["clust"] = adata.obs[name].astype("category")
    adata.obs["y"] = adata.obs["clust"].cat.codes
    if adata.obs["y"].isnull().any() or adata.obs["clust"].isnull().any():
        warn("Some or all cells not assigned to cluster.")
    adata.uns["num_clusts"] = adata.obs["clust"].cat.categories.size
    clusterindices = {}
    for k in range(adata.uns["num_clusts"]):
        try:
            clusterindices[k] = (adata.obs["y"] == k).to_numpy().nonzero()[0]
        except AttributeError:
            clusterindices[k] = (adata.obs["y"] == k).nonzero()[0]
    adata.uns["clusterindices"] = clusterindices
    return adata
