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

from anndata import AnnData
from scipy.sparse.linalg import svds
import numpy as np


def pca(data, dim=3, center=True, copy=False):
    """Runs PCA

    Args
    ----
    data: anndata.AnnData
        input data
    dim: int, optional
        number of PCs to compute
    center: bool, optional
        determines whether to center data before running PCA
    copy: 
        determines whether a copy of `AnnData` object is returned

    Returns
    -------
    anndata.AnnData
        object with ``obsm["X_pca"]``, and ``varm["PCs"]`` set
    """

    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        Xcent, pcs, Xpca = pca(adata.X, dim=dim, center=center)
        adata.obsm["X_pca"] = Xpca
        adata.varm["PCs"] = pcs
        return adata
    Xcent = data.copy()
    if center:
        Xcent -= data.mean(axis=0)[np.newaxis, :]
    pcs = svds(Xcent.T, dim, return_singular_vectors="u")[0][:, ::-1]
    Xpca = Xcent.dot(pcs)
    return (Xcent, pcs, Xpca)
