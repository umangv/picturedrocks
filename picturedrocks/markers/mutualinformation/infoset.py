# Copyright © 2018 Anna Gilbert, Alexander Vargo, Umang Varma
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

from logging import info

import numba as nb
import numpy as np
import scipy.sparse


def makeinfoset(adata, include_y):
    """Discretize data

    Args
    ----
    adata: anndata.AnnData
        The data to discretize. By default data is discretized as
        `round(log2(X + 1))`.
    include_y: bool
        Determines if the `y` (cluster label) column in included in the 
        `InformationSet` object
    
    Returns
    -------
    picturedrocks.markers.mutualinformation.infoset.InformationSet
        An object that can be used to perform information theoretic
        calculations.
    """
    # we currently don't support scipy sparse matrices
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    X = np.log2(X + 1).round().astype(int)
    if include_y:
        y = adata.obs["y"]
        X = np.concatenate([X, y[:, None]], axis=1)
    return InformationSet(X, include_y)


@nb.jitclass(
    [
        ("has_y", nb.bool_),
        ("X", nb.int64[:, :]),
        ("N", nb.int64),
        ("P", nb.int64),
        ("_shift", nb.int64),
    ]
)
class InformationSet:
    """Stores discrete gene expression matrix

    Args
    ----
    X: numpy.ndarray
        a (num_obs, num_vars) shape array with ``dtype`` :class:`int`
    has_y: bool
        whether the array `X` has a target label column (a `y` column) as its
        last column
    """

    def __init__(self, X, has_y=False):
        self.has_y = has_y
        self.X = X
        self.N = self.X.shape[0]
        self.P = self.X.shape[1]
        self._shift = int(np.log2(self.X.max()) + 1)

    def entropy(self, cols):
        """Entropy of an ensemble of columns
        
        Args
        ----
        cols: numpy.ndarray
            a 1-d array (of dtype int64) with indices of columns to compute
            entropy over
        
        Returns
        -------
        numpy.int64
            the Shannon entropy of `cols`
        """
        n_cols = len(cols)
        delta_prob = 1 / self.N
        mat = self.X[:, cols]
        counts = np.zeros((2 ** self._shift) ** n_cols)
        for i in range(self.N):
            ind = 0
            for j in range(n_cols):
                ind = ind << self._shift
                ind = ind | mat[i, j]
            counts[ind] += delta_prob
        h = 0
        for e in counts:
            if e > 0:
                h += -e * np.log2(e)
        return h

    def entropy_wrt(self, cols):
        """Compute multiple entropies at once

        This method computes the entropy of `cols + [i]` iterating over all
        possible values of `i` and returns an array of entropies (one for
        each column)

        Args
        ----
        cols: numpy.ndarray
            a 1-d array of columns

        Returns
        -------
        numpy.ndarray
            a 1-d array of entropies (where entry `i` corresponds to the
            entropy of columns `cols` together with column `i`)
        """
        n_feats = self.P - 1 if self.has_y else self.P
        if len(cols) > 0:
            cols_template = np.concatenate((cols, np.array([0])))
        else:
            cols_template = np.array([0], dtype=np.int64)

        def entropy_wrt_i(i):
            cols_template[-1] = i
            return self.entropy(cols_template)

        hs = np.array([entropy_wrt_i(i) for i in range(n_feats)])
        return hs
