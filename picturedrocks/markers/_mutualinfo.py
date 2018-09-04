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

import datetime
from logging import info
from abc import ABC, abstractmethod

import numba as nb
import numpy as np
import scipy.sparse


def makeinfoset(adata, include_y):
    """Discretize data"""
    # we currently don't support scipy sparse matrices
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    X = np.log2(X + 1).round().astype(int)
    if include_y:
        y = adata.obs["y"]
        X = np.concatenate([X, y], axis=1)
    return InformationSet(X, include_y)


def mutualinfo(infoset, n, pool=None, obj="mrmr"):
    """Compute markers using mutual information

    Args
    ----
    infoset: picturedrocks.markers.InformationSet
        discrete gene expression matrix
    n: int
        number of markers to select
    pool: list, optional
        pool of genes to restrict marker selection search 
    obj: {"mrmr", "mim", "cife", "jmi", "deg3"}
        objective function to use
    
    Returns
    -------
    list
        ``n`` element list of markers
    """

    assert infoset.hasy, "infoset must include target (cluster) labels"

    if pool is None:
        pool = range(infoset.X.shape[1] - 1)

    start = datetime.datetime.now()
    S = []
    Sset = set()
    H = infoset.H
    for m in range(n):
        maxobj = float("-inf")
        argmaxobj = -1
        if obj == "mim" or m == 0:
            # all objectives do the same thing when m = 0, but some
            # objectives simplify greatly m is not 0, so handle that here
            for i in pool:
                if i in Sset:
                    continue
                curobj = H((i,)) - H((-1, i))
                maxobj, argmaxobj = max((maxobj, argmaxobj), (curobj, i))
        elif obj == "mrmr":
            for i in pool:
                if i in Sset:
                    continue
                curobj = -m * H((-1, i))  # infoset.Iwrtx([-1], i, 2)
                for x in S:
                    curobj += H((x, i))  # (1.0/m) * infoset.Iwrtx([x], i, 3)
                maxobj, argmaxobj = max((maxobj, argmaxobj), (curobj, i))
        elif obj == "jmi":
            for i in pool:
                if i in Sset:
                    continue
                curobj = 0
                for x in S:
                    curobj += H((x, i)) - H((-1, x, i))
                maxobj, argmaxobj = max((maxobj, argmaxobj), (curobj, i))
        elif obj == "cife":
            for i in pool:
                if i in Sset:
                    continue
                curobj = (1 - m) * (H((i,)) - H((-1, i)))
                for x in S:
                    curobj += H((x, i)) - H((-1, x, i))
                maxobj, argmaxobj = max((maxobj, argmaxobj), (curobj, i))
        elif obj == "deg3":
            mchoose2 = (m * (m - 1)) // 2
            for i in pool:
                if i in Sset:
                    continue
                curobj = (1 - m + mchoose2) * (H((i,)) - H((-1, i)))
                for x in S:
                    curobj += (2 - m) * (H((x, i)) - H((-1, x, i)))
                for x1, x2 in combinations(S, 2):
                    curobj += H((x1, x2, i)) - H((-1, x1, x2, i))
                maxobj, argmaxobj = max((maxobj, argmaxobj), (curobj, i))
        else:
            raise ValueError("Unknown objective function.")
        # common for all objectives
        S.append(argmaxobj)
        Sset.add(argmaxobj)
        info("Features: {}".format(repr(S)))
    end = datetime.datetime.now()
    timedelta = end - start
    info(
        "It took {:.2f} minutes to find {} features via {}.".format(
            timedelta.total_seconds() / 60, n, obj
        )
    )
    return S


class IterativeFeatureSelection(ABC):
    @abstractmethod
    def __init__(self, infoset):
        self.infoset = infoset
        self.score = None
        self.S = []

    @abstractmethod
    def add(self, ind):
        pass

    @abstractmethod
    def remove(self, ind):
        pass

    def autoselect(self, n_feats):
        for i in range(n_feats):
            best = np.argmax(self.score)
            self.add(best)


class CIFE(IterativeFeatureSelection):
    def __init__(self, infoset):
        assert infoset.has_y, "Information Set must have target labels"
        self.infoset = infoset
        self.S = []
        self.base_score = (
            self.infoset.entropy_wrt(np.arange(0))
            + self.infoset.entropy(np.array([-1]))
            - self.infoset.entropy_wrt(np.array([-1]))
        )
        self.penalty = np.zeros(len(self.base_score))
        self.score = self.base_score[:]

    def add(self, ind):
        self.S.append(ind)
        penalty_delta = (
            self.base_score
            + self.infoset.entropy(np.array([ind]))
            - self.infoset.entropy_wrt(np.array([ind]))
            - self.infoset.entropy(np.array([-1, ind]))
            + self.infoset.entropy_wrt(np.array([-1, ind]))
        )
        self.penalty += penalty_delta
        self.score = self.base_score - self.penalty
        self.score[self.S] = float("-inf")
    
    def remove(self, ind):
        raise NotImplementedError("Remove has not been implemented yet")

class MIM(IterativeFeatureSelection):
    def __init__(self, infoset):
        assert infoset.has_y, "Information Set must have target labels"
        self.infoset = infoset
        self.S = []
        self.base_score = (
            self.infoset.entropy_wrt(np.arange(0))
            + self.infoset.entropy(np.array([-1]))
            - self.infoset.entropy_wrt(np.array([-1]))
        )
        self.score = self.base_score.copy()

    def add(self, ind):
        self.S.append(ind)
        self.score = self.base_score.copy()
        self.score[self.S] = float("-inf")

    def remove(self, ind):
        raise NotImplementedError("Remove has not been implemented yet")

    def autoselect(self, n_feats):
        nbest = np.argpartition(self.score, -n_feats)[-n_feats:]
        nbest = nbest[np.argsort(self.score[nbest])[::-1]]
        self.S.extend(nbest)
        self.score[self.S] = float("-inf")
        return nbest

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
