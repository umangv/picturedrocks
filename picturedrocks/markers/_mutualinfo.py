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

import numpy as np
import scipy.sparse
import datetime
from logging import info


def makeinfoset(adata):
    """Discretize data"""
    # we currently don't support scipy sparse matrices
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    return InformationSet(np.log2(X + 1).round().astype(int), adata.obs["y"])


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


class InformationSet:
    """Stores discrete gene expression matrix

    Args
    ----
    X: numpy.ndarray
        a (num_obs, num_vars) shape array with ``dtype`` :class:`int`
    y: numpy.ndarray, optional
        a (num_obs,) shape array
    """
    def __init__(self, X, y=None):
        self.hasy = not y is None
        # insert a column of zeros; this is our baseline (i.e., it should have a
        # score of zero always)
        X = np.c_[X, np.zeros((X.shape[0], 1))]
        self.baseline_index = -1
        if self.hasy:
            self.X = np.c_[X, y]
            self.baseline_index -= 1
        else:
            self.X = X
        self.N = X.shape[0]
        self._H = {}
        self._shift = int(np.log2(X.max()) + 1)

    def cols(self, cols):
        """Ensemble of columns
        
        This function discretizes the ensemble of columns into
        one column. 
        """
        ncols = len(cols)
        return self.X[:, cols].dot(2 ** (self._shift * np.arange(ncols - 1, -1, -1)))

    def H(self, cols):
        """Entropy of an ensemble of columns
        
        Returns `H(cols)`, where `cols` is treated as an ensemble.
        """
        if cols not in self._H:
            vals, counts = np.unique(self.cols(cols), return_counts=True)
            p = counts / self.N
            self._H[cols] = np.sum(-p * np.log2(p))
        return self._H[cols]

    def Iapprox(self, cols, deg):
        """Approximate multivariate MI
        
        If `cols = [x_1, ..., x_m]`, this function returns the
        degree `deg` approximation of `I(x_1; ...; x_m)`.
        """
        ret = 0
        for k in range(deg):
            for U in combinations(cols, k + 1):
                ret += ((-1) ** k) * self.H(U)
        return ret

    def Iwrtx(self, S, x, deg):
        """Approximate multivariate MI with resp to x
        
        If `S = [x_1, x_2, ..., x_m]`, this fuction returns the
        sum of the terms in the inclusion/exclusion expansion for
        `I(x_1; x_2; ...; x_m; x)` of degree `deg` that involve `x`.
        
        This is a faster substitute for Iapprox for calculations where
        only the terms involve `x` are required (e.g., to compute
        `argmax_x (x_1;...;x_n; x)`
        """
        ret = 0
        for k in range(deg):
            for U in combinations(S, k):
                ret += ((-1) ** k) * self.H(U + (x,))
        return ret
