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

import numpy as np
import plotly.graph_objs as go
from scanpy.preprocessing import normalize_per_cell, log1p
import scipy.spatial.distance
from scipy.sparse import issparse
from anndata import AnnData

from .read import process_clusts


def kfoldindices(n, k, random=False):
    """Generate indices for k-fold cross validation

    Args
    ----
    n: int
        number of observations
    k: int
        number of folds
    random: bool
        determines whether to randomize the order

    Yields
    ------
    numpy.ndarray
        array of indices in each fold
    """
    basearray = np.arange(n)
    if random:
        np.random.shuffle(basearray)
    lengthfloor = n // k
    extra = n % k
    cur = 0
    while cur < n:
        thislength = lengthfloor + 1 if extra > 0 else lengthfloor
        yield basearray[cur : cur + thislength]
        cur += thislength
        extra -= 1  # this should be extra = max(extra - 1, 0),
        #            but it doesn't matter


class PerformanceReport:
    """Report actual vs predicted statistics
    
    Args
    ----
    y: numpy.ndarray
        actual cluster labels, (N, 1)-shaped numpy array
    yhat: numpy.ndarray
        predicted cluster labels, (N, 1)-shaped numpy array
    """

    def __init__(self, y, yhat):
        self.y = y
        self.yhat = yhat
        self.N = y.shape[0]

        self.K = y.max() + 1
        assert np.equal(
            np.unique(self.y), range(self.K)
        ).all(), "Cluster labels should be 0, 1, 2, ..., K -1"

        self.clusterindices = {}
        self._genclusterindices()

    def _genclusterindices(self):
        """Compute and store indices for cells in each cluster."""
        for k in range(self.K):
            self.clusterindices[k] = np.nonzero(self.y == k)[0]
        self.nk = np.array([len(self.clusterindices[k]) for k in range(self.K)])
        # nk[k] is the number of entries in cluster k

    def wrong(self):
        """Returns the number of cells misclassified."""
        return np.sum((self.y.flatten() != self.yhat) * 1.0)

    def printscore(self):
        """Print a message with the score"""
        wrong = self.wrong()
        print(
            "{} out of {} incorrect: {:.2f}%".format(
                wrong, self.N, 100 * wrong / self.N
            )
        )

    def getconfusionmatrix(self):
        """Get the confusion matrix for the latest run
        
        Returns
        -------
        numpy.ndarray
            array of shape (K, K), with the [i, j] entry being the fraction
            of cells in cluster i that were predicted to be in cluster j
        """
        K = self.K
        freq_table = np.zeros([K, K])
        for i in range(K):
            clust, clust_count = np.unique(
                self.yhat[self.clusterindices[i]], return_counts=True
            )
            for j, k in enumerate(clust):
                freq_table[i, k] = clust_count[j] / self.nk[i]
        return freq_table

    def confusionmatrixfigure(self):
        """Compute and make a confusion matrix figure
        
        Returns
        -------
        `plotly figure`
            confusion matrix 
        """
        freq_table = self.getconfusionmatrix()
        shape = freq_table.shape
        trace = go.Heatmap(
            z=freq_table,
            x=np.arange(shape[1]),
            y=np.arange(shape[0]),
            colorscale="Greys",
            reversescale=True,
        )
        layout = go.Layout(
            title="Confusion Matrix",
            xaxis=dict(title="Predicted Cluster"),
            yaxis=dict(title="Actual Cluster", scaleanchor="x"),
            width=450,
            height=450,
            margin=go.layout.Margin(l=70, r=70, t=70, b=70, pad=0, autoexpand=False),
            annotations=[
                dict(
                    text="Rows sum to 1",
                    x=0.5,
                    y=1,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                )
            ],
        )
        return go.Figure(data=[trace], layout=layout)

    def show(self):
        """Print a full report
        
        This uses `iplot`, so we assume this will only be run in a Jupyter
        notebook and that `init_notebook_mode` has already been run.
        """
        from plotly.offline import iplot

        self.printscore()
        iplot(self.confusionmatrixfigure())


class FoldTester:
    """Performs K-fold Cross Validation for Marker Selection

    :class:`FoldTester` can be used to evaluate various marker selection
    algorithms. It can split the data in `K` folds, run marker selection
    algorithms on these folds, and classify data based on testing and
    training data.

    Args
    ----
    adata: anndata.AnnData
        data to slice into folds
    """

    def __init__(self, adata):
        self.adata = adata

        self.folds = None
        self.yhat = None
        self.markers = None

    def makefolds(self, k=5, random=False):
        """Makes folds

        Args
        ----
        k: int
            the value of K
        random: bool
            If true, `makefolds` will make folds randomly. Otherwise, the
            folds are made in order (i.e., the first ``ceil(N / k)`` cells in
            the first fold, etc.)
        """
        self.folds = list(kfoldindices(self.adata.n_obs, k, random))

    def savefolds(self, file):
        """Save folds to a file

        Args
        ----
        file: str
            filename to save (typically with a ``.npz`` extension)
        """
        d = {"k": len(self.folds), "y": self.adata.obs["y"].values}
        for i, f in enumerate(self.folds):
            d["fold{}".format(i)] = f
        return np.savez(file, **d)

    def loadfolds(self, file):
        """Load folds from a file

        The file can be one saved either by :meth:`FoldTester.savefolds` or
        :meth:`FoldTester.savefoldsandmarkers`. In the latter case, it will
        not load any markers.

        See Also
        --------
        :meth:`FoldTester.loadfoldsandmarkers`.
        """
        d = np.load(file)
        k = d["k"]
        self.folds = [d["fold{}".format(i)] for i in range(k)]
        assert np.array_equal(
            self.adata.obs["y"], d["y"].ravel()
        ), "y vector does not match."
        assert self.validatefolds(), "folds are not partition of indices"

    def validatefolds(self):
        """Ensure that all observations are in exactly one fold
        
        Returns
        -------
        bool
        """
        counts = np.zeros(self.adata.n_obs)
        for f in self.folds:
            counts[f] += 1
        return np.alltrue(counts == 1)

    def selectmarkers(self, select_function):
        """Perform a marker selection algorithm on each fold

        Args
        ----
        select_function: function
            a function that takes in an :class:`AnnData <anndata.AnnData>`
            object and outputs a list of gene markers, given by their index 

        Note
        ----
        The `select_function` should not attempt to modify data in-place.  Any
        preprocessing should be done on a copy. 
        """
        k = len(self.folds)
        self.markers = []
        for f in self.folds:
            mask = np.zeros(self.adata.n_obs, dtype=bool)
            mask[f] = True
            traindata = self.adata[~mask, :]
            self.markers.append(select_function(traindata))

    def classify(self, classifier):
        """Classify each cell using training data from other folds

        For each fold, we project the data onto the markers selected for that
        fold, which we treat as test data. We also project the complement of
        the fold and treat that as training data.

        Args
        ----
        classifier
            a classifier that trains with a training data set and predicts
            labels of test data. See `NearestCentroidClassifier` for an
            example.

        Note
        ----
        The `classifier` should not attempt to modify data in-place.  Any
        preprocessing should be done on a copy. 
        """
        self.yhat = np.zeros(self.adata.n_obs, dtype=int) - 1
        for i, f in enumerate(self.folds):
            mask = np.zeros(self.adata.n_obs, dtype=bool)
            mask[f] = True
            traindata = self.adata[~mask, :][:, self.markers[i]]
            c = classifier()
            c.train(traindata)
            self.yhat[f] = c.test(self.adata.X[f, :][:, self.markers[i]])

    def savefoldsandmarkers(self, file):
        """Save folds and markers for each fold

        This saves folds, and for each fold, the markers previously found by
        :meth:`FoldTester.selectmarkers`.

        Args
        ----
        file: str
            filename to save to (typically with a ``.npz`` extension)
        """
        d = {"k": len(self.folds), "y": self.adata.obs["y"]}
        for i, f in enumerate(self.folds):
            d["fold{}".format(i)] = f
        for i, m in enumerate(self.markers):
            d["marker{}".format(i)] = m
        return np.savez(file, **d)

    def loadfoldsandmarkers(self, file):
        """Load folds and markers

        Loads a folds and markers file saved by
        :meth:`FoldTester.savefoldsandmarkers`

        Args
        ----
        file: str
            filename to load from (typically with a ``.npz`` extension)
        
        See Also
        --------
        :meth:`FoldTester.loadfolds`
        """
        d = np.load(file)
        k = d["k"]
        self.folds = [d["fold{}".format(i)] for i in range(k)]
        self.markers = [d["marker{}".format(i)] for i in range(k)]
        assert np.array_equal(
            self.adata.obs["y"], d["y"].ravel()
        ), "y vector does not match."
        assert self.validatefolds(), "folds are not partition of indices"


def truncatemarkers(ft, n_markers):
    ftnew = FoldTester(ft.adata)
    ftnew.folds = ft.folds
    ftnew.markers = [m[:n_markers] for m in ft.markers]
    return ftnew


def merge_markers(ft, n_markers):
    ftnew = FoldTester(ft.adata)
    ftnew.folds = ft.folds
    ftnew.markers = [
        list(set(np.array(markers)[:, :n_markers].flatten())) for markers in ft.markers
    ]
    return ftnew


class NearestCentroidClassifier:
    """Nearest Centroid Classifier for Cross Validation

    Computes the centroid of each cluster label in the training data, then
    predicts the label of each test data point by finding the nearest centroid.
    """

    def __init__(self):
        self.xkibar = None

    def train(self, adata):
        adata = adata.copy()
        adata.X = _toarray(adata.X)
        normalize_per_cell(adata, 1000, min_counts=0)
        log1p(adata)
        adata = process_clusts(adata)
        self.xkibar = np.array(
            [
                adata.X[adata.uns["clusterindices"][k]].mean(axis=0).tolist()
                for k in range(adata.uns["num_clusts"])
            ]
        )

    def test(self, Xtest):
        testdata = AnnData(Xtest)
        normalize_per_cell(testdata, 1000, min_counts=0)
        log1p(testdata)
        testdata.X = _toarray(testdata.X)
        dxixk = scipy.spatial.distance.cdist(testdata.X, self.xkibar)
        return dxixk.argmin(axis=1)

def _toarray(arr_like):
    if issparse(arr_like):
        return arr_like.toarray()
    return arr_like
