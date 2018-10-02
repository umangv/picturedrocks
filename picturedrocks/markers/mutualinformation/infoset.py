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
    Union[InformationSet, SparseInformationSet]
        An object that can be used to perform information theoretic
        calculations.
    """
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.copy()
        X.data = np.log2(X.data + 1).round().astype(int)
        if include_y:
            y = adata.obs["y"]
            X = scipy.sparse.hstack([X, y[:, None]])
        return SparseInformationSet(X, include_y)
    else:
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


class SparseInformationSet:
    """Stores sparse discrete gene expression matrix

    Args
    ----
    X: scipy.sparse.csc_matrix
        a (num_obs, num_vars) shape matrix with ``dtype`` :class:`int`
    has_y: bool
        whether the array `X` has a target label column (a `y` column) as its
        last column
    """

    def __init__(self, X, has_y=False):
        self.has_y = has_y
        # our algorithm uses csc matrices under the assumption and zeros are
        # eliminated and entries have been sorted. Ensure this is the case.
        self.X = scipy.sparse.csc_matrix(X)
        self.X.eliminate_zeros()
        self.X.sort_indices()
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
        mcol = _sparse_make_master_col(self.X, cols, self._shift)
        return _sparse_entropy(
            mcol.indices, mcol.data, self.N, 2 ** (self._shift * (len(cols) + 1))
        )

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
        mcol = _sparse_make_master_col(self.X, cols, self._shift)
        if not np.issubdtype(mcol.dtype, np.integer):
            import pdb

            pdb.set_trace()
        return _sparse_entropy_wrt(
            self.X.indices,
            self.X.indptr,
            self.X.data,
            mcol.indices,
            mcol.data,
            self.N,
            n_feats,
            len(cols),
            self._shift,
        )

    def todense(self):
        """Return a InformationSet with the same data stored as a dense array"""
        return InformationSet(self.X.toarray(), self.has_y)

    @staticmethod
    def fromdense(denseinfoset):
        """Create SparseInformationSet with the same data stored as a sparse matrix"""
        return SparseInformationSet(denseinfoset.X, denseinfoset.has_y)


@nb.njit
def _sparse_entropy_wrt(
    dindices, dindptr, ddata, mindices, mdata, n_rows, n_feats, n_mcols, shift
):
    mdata = mdata << shift

    def entropy_wrt_i(cindices, cdata, i):
        counts = np.zeros(2 ** (shift * (n_mcols + 1)), dtype=np.int64)
        counts[0] = n_rows
        cur_cind = 0
        cur_mind = 0
        # while at least one of current/master column has remaining entries to visit
        while cur_cind < cindices.size or cur_mind < mindices.size:
            # either we've run out of entries in the master column, or the
            # current column's next entry is smaller
            if cur_mind >= mindices.size or (
                cur_cind < cindices.size and cindices[cur_cind] < mindices[cur_mind]
            ):
                curval = cdata[cur_cind]
                cur_cind += 1
            # either we've run out of entries in the current column, or the
            # master column's next entry is smaller
            elif cur_cind >= cindices.size or cindices[cur_cind] > mindices[cur_mind]:
                curval = mdata[cur_mind]
                cur_mind += 1
            # neither column is exhausted and row indices are equal for both columns
            else:
                curval = cdata[cur_cind] + mdata[cur_mind]
                cur_cind += 1
                cur_mind += 1
            counts[0] -= 1
            counts[curval] += 1
        h = 0
        counts = counts / n_rows
        for e in counts:
            if e > 0:
                h += -e * np.log2(e)
        return h

    hs = np.array(
        [
            entropy_wrt_i(
                dindices[dindptr[i] : dindptr[i + 1]],
                ddata[dindptr[i] : dindptr[i + 1]],
                i,
            )
            for i in range(n_feats)
        ]
    )
    return hs


@nb.njit
def _sparse_entropy(indices, data, n_rows, max_val):
    counts = np.zeros(max_val + 1, dtype=np.int64)
    counts[0] = n_rows
    cur_ind = 0
    while cur_ind < indices.size:
        curval = data[cur_ind]
        cur_ind += 1
        counts[0] -= 1
        counts[curval] += 1
    h = 0
    counts = counts / n_rows
    for e in counts:
        if e > 0:
            h += -e * np.log2(e)
    return h


# I have some code that is very likely much faster than this, but I have not
# tested it. Leaving the cleaner but slower code here.
def _sparse_make_master_col(X, cols, shift):
    if len(cols) > 0:
        return X[:, cols] @ scipy.sparse.csr_matrix(
            2 ** (shift * np.arange(len(cols)))[:, None]
        )
    else:
        return scipy.sparse.csc_matrix((X.shape[0], 1), dtype=int)
