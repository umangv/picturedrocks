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

import collections

import colorlover as cl
import numpy as np
import plotly.graph_objs as go
import scipy.sparse
from anndata import AnnData
from umap import UMAP

try:
    from scanpy.preprocessing.simple import pca
except ModuleNotFoundError:
    from scanpy.preprocessing import pca


def _deep_merge_dict(source, destination):
    """Merges dict-like objects in a deep manner

    e.g., if source = {'a': {'b': 5}, 'c': 4} and destination = {'a': {'b':
    1, 'bb': 0},'d': 0} then destination is changed **in place** to {'a':
    {'b': 5, 'bb': 0}, 'c': 4, 'd': 0}
    """

    for key in source.keys():
        if isinstance(source[key], collections.Mapping) and key in destination:
            _deep_merge_dict(source[key], destination[key])
        else:
            destination[key] = source[key]


def _scatter(coords, *args, **kwargs):
    """Run the appropriate scatter function"""
    assert coords.shape[1] in [2, 3], "incorrect dimensions for coords"
    if coords.shape[1] == 2:
        try:
            return go.Scattergl(x=coords[:, 0], y=coords[:, 1], *args, **kwargs)
        except AttributeError:
            return go.Scatter(x=coords[:, 0], y=coords[:, 1], *args, **kwargs)
    else:
        return go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], *args, **kwargs
        )


def genericplot(celldata, coords, **scatterkwargs):
    """Generate a figure for some embedding of data

    This function supports both 2D and 3D plots. This may be used to plot
    data for any embedding (e.g., PCA or t-SNE). For example usage, see code
    for `pcafigure`.

    Args
    ----
    celldata: anndata.AnnData
        data to plot
    coords: numpy.ndarray
        (N, 2) or (N, 3) shaped coordinates of the embedded data 
    **scatterkwargs
        keyword arguments to pass to ``Scatter`` or ``Scatter3D`` in `plotly`
        (dictionaries are merged recursively)
    """

    clusterindices = celldata.uns["clusterindices"]
    colscal = cl.scales["9"]["qual"]["Set1"]

    plotdata = [
        _scatter(
            coords[inds],
            mode="markers",
            marker=dict(
                size=2,
                color=colscal[k % len(colscal)],  # set color to an array/list
                #                                  of desired values
                opacity=1,
            ),
            name=celldata.obs["clust"].cat.categories[k],
            hoverinfo="name",
        )
        for k, inds in clusterindices.items()
    ]

    if scatterkwargs:
        for p in plotdata:
            _deep_merge_dict(scatterkwargs, p)

    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0), hovermode="closest")
    return go.Figure(data=plotdata, layout=layout)


def pcafigure(celldata, **scatterkwargs):
    """Make a 3D PCA figure for an AnnData object

    Args
    ----
    celldata: anndata.AnnData
        data to plot
    **scatterkwargs
        keyword arguments to pass to ``Scatter`` or ``Scatter3D`` in `plotly`
        (dictionaries are merged recursively)
    """
    if "X_pca" not in celldata.obsm_keys() or celldata.obsm["X_pca"].shape[1] < 3:
        print("Need 3 PCs. Calculating now.")
        pca(celldata, 3)
    return genericplot(celldata, celldata.obsm["X_pca"][:, :3], **scatterkwargs)


def umapfigure(adata, **scatterkwargs):
    if "X_umap" not in adata.obsm_keys():
        if "X_pca" not in adata.obsm_keys() or adata.obsm["X_pca"].shape[1] < 30:
            pca(adata, 30, zero_center=not scipy.sparse.issparse(adata.X))
        umap = UMAP()
        adata.obsm["X_umap"] = umap.fit_transform(adata.obsm["X_pca"][:, :30])
    return genericplot(adata, adata.obsm["X_umap"], **scatterkwargs)


def genericwrongplot(celldata, coords, yhat, labels=None, **scatterkwargs):
    """Plot figure with incorrectly classified points highlighted
    
    This can be used with any 2D or 3D embedding (e.g., PCA or t-SNE). For
    example code, see `pcawrongplot`.

    Args
    ----
    celldata: anndata.AnnData
        data to plot
    coords: numpy.ndarray
        (N, 2) or (N, 3) shaped array with coordinates to plot
    yhat: numpy.ndarray
        (N, 1) shaped array of predicted `y` values
    labels: list, optional
        list of axis titles
    **scatterkwargs
        keyword arguments to pass to ``Scatter`` or ``Scatter3D`` in `plotly`
        (dictionaries are merged recursively)
    """

    y = celldata.obs["y"]
    if yhat.shape == (y.shape[0], 1):
        yhat = yhat[:, 0]
    assert yhat.shape == y.shape, "yhat must have shape (N, 1)"

    colscal = cl.scales["9"]["qual"]["Set1"]

    # indices (by cluster) where wrong
    wronginds = {}
    for k in range(celldata.uns["num_clusts"]):
        wronginds[k] = np.nonzero((y == k) & np.not_equal(yhat, y))[0]

    # indices (by cluster) where correct
    clustinds = {}
    for k in range(celldata.uns["num_clusts"]):
        clustinds[k] = np.nonzero((y == k) & np.equal(yhat, y))[0]

    # Get the points that are wrong
    plotdata = [
        _scatter(
            coords[inds],
            mode="markers",
            marker=dict(size=2, color=colscal[k % len(colscal)], opacity=1),
            name="Cluster {}".format(k),
            hoverinfo="name+text",
            text=["Predict {}".format(str(a)) for a in yhat[inds]],
        )
        for k, inds in wronginds.items()
    ] + [
        _scatter(
            coords[inds],
            mode="markers",
            marker=dict(size=2, color=colscal[k % len(colscal)], opacity=0.2),
            name="Cluster {}".format(k),
            hoverinfo="name",
            showlegend=False,
        )
        for k, inds in clustinds.items()
    ]

    if scatterkwargs:
        for p in plotdata:
            _deep_merge_dict(scatterkwargs, p)

    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0), hovermode="closest")
    if labels:
        layout["scene"] = dict(xaxis=dict(title=labels[0]), yaxis=dict(title=labels[1]))
        if len(labels) == 3:
            layout["scene"]["zaxis"] = dict(title=labels[2])

    return go.Figure(data=plotdata, layout=layout)


def pcawrongplot(celldata, yhat, **scatterkwargs):
    """Generate a 3D PCA figure with incorrectly classified points highlighted

    Args
    ----
    celldata: anndata.AnnData
        data to plot
    yhat: numpy.ndarray
        (N, 1) shaped array of predicted `y` values
    **scatterkwargs
        keyword arguments to pass to ``Scatter`` or ``Scatter3D`` in `plotly`
        (dictionaries are merged recursively)
    """

    if "X_pca" not in celldata.obsm_keys() or celldata.obsm["X_pca"].shape[1] < 3:
        print("Need 3 PCs. Calculating now.")
        pca(celldata, 3)
    return genericwrongplot(
        celldata,
        celldata.obsm["X_pca"][:, :3],
        yhat,
        labels=["PC1", "PC2", "PC3"],
        **scatterkwargs
    )


def plotgeneheat(celldata, coords, genes, hide_clusts=False, **scatterkwargs):
    """Generate gene heat plot for some embedding of AnnData

    This generates a figure with multiple dropdown options. The first option
    is "Clust" for a plot similar to `genericplot`, and the remaining
    dropdown options correspond to genes specified in `genes`. When
    `celldata.genes` is defined, these drop downs are labeled with the gene
    names.

    Args
    ----
    celldata: anndata.AnnData
        data to plot
    coords: numpy.ndarray
        (N, 2) or (N, 3) shaped coordinates of the embedded data (e.g., PCA
        or tSNE)
    genes: list
        list of gene indices or gene names
    hide_clusts: bool
        Determines if cluster labels are ignored even if they are available
    """

    clustscal = cl.scales["9"]["qual"]["Set1"]
    genescal = np.array(cl.scales["8"]["seq"]["Blues"])
    if (not hide_clusts) and "y" in celldata.obs_keys():
        first_label = "Clust"
        numclusts = celldata.uns["num_clusts"]
        clusterindices = celldata.uns["clusterindices"]

        plotbyclust = [
            _scatter(
                coords[inds],
                mode="markers",
                marker=dict(
                    size=4,
                    color=clustscal[k % len(clustscal)],  # set color to an array/list
                    #                                  of desired values
                ),
                name=celldata.obs["clust"].cat.categories[k],
                hoverinfo="name",
            )
            for k, inds in clusterindices.items()
        ]
    else:
        first_label = "Cells"
        numclusts = 1
        plotbyclust = [
            _scatter(coords, mode="markers", marker=dict(size=4), name="cells")
        ]

    genes = [
        g if isinstance(g, (int, np.integer)) else celldata.var.index.get_loc(g)
        for g in genes
    ]
    genenames = celldata.var.index[genes].tolist()

    geneexpr = celldata.X[:, genes]
    exprnorm = np.round(geneexpr * 7 / geneexpr.max(axis=0)).astype(
        int
    )  # this sets the color of each point
    numgenes = geneexpr.shape[1]

    plotbygene = [
        _scatter(
            coords,
            mode="markers",
            marker=dict(size=4, color=genescal[exprnorm[:, i]]),
            name=genenames[i],
            text=geneexpr[:, i].astype(str),
            hoverinfo="name+text",
            visible=False,
        )
        for i, genename in enumerate(genenames)
    ]
    buttons = [
        dict(
            label=first_label,
            method="update",
            args=[
                {
                    "visible": [True] * numclusts + [False] * numgenes,
                    "showlegend": True,
                },
                {},
            ],
        )
    ]
    for i, genename in enumerate(genenames):
        v = [False] * (numclusts + numgenes)
        v[numclusts + i] = True
        buttons.append(
            dict(
                label=genename,
                method="update",
                args=[{"visible": v, "showlegend": False}, {}],
            )
        )
    if scatterkwargs:
        for p in plotbyclust + plotbygene:
            _deep_merge_dict(scatterkwargs, p)

    updatemenus = [
        dict(
            buttons=buttons,
            direction="down",
            x=0.1,
            y=1.1,
            xanchor="left",
            yanchor="top",
            pad={"r": 10, "t": 10},
        )
    ]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        updatemenus=updatemenus,
        annotations=[
            dict(
                text="View:",
                x=0.01,
                y=1.065,
                xref="paper",
                yref="paper",
                align="left",
                showarrow=False,
            )
        ],
        showlegend=True,
        legend_orientation="h"
    )

    return go.Figure(data=plotbyclust + plotbygene, layout=layout)
