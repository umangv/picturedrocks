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

from anndata import AnnData
import colorlover as cl
import plotly.graph_objs as go

def genericplot(celldata, coords):
    """Generate a figure for some embedding of Rocks data

    This function supports both 2D and 3D plots. This may be used to plot data
    for any embedding (e.g., PCA or t-SNE). For example usage, see code for
    `pcafigure`.

    :param celldata: an AnnData object
    :param coords: (N, 2) or (N, 3) shaped coordinates of the embedded data 
    """

    def scatter(coords, *args, **kwargs):
        """Run the appropriate scatter function"""
        assert coords.shape[1] in [2,3], "incorrect dimensions for coords"
        if coords.shape[1] == 2:
            return go.Scatter(x=coords[:,0], y=coords[:,1], *args, **kwargs)
        else:
            return go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2],
                    *args, **kwargs)
    clusterindices = celldata.uns['clusterindices']
    colscal = cl.scales['9']['qual']['Set1']

    plotdata = [scatter(
            coords[inds],
            mode='markers',
            marker=dict(
                size=4,
                color=colscal[k % len(colscal)], # set color to an array/list
                #                                  of desired values
                opacity=1),
            name=celldata.obs['clust'].cat.categories[k],
            hoverinfo="name"
            )
            for k, inds in clusterindices.items()]

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
    )
    return go.Figure(data=plotdata, layout=layout)

def pcafigure(celldata):
    """Make a 3D PCA figure for a Rocks object

    :param celldata: an AnnData object
    """
    if celldata.obsm['X_pca'] is None# TODO: or celldata.Xpca.shape[1] < 3:
        #print("Need 3 PCs. Calculating now.")
        pca(celldata, 3)
    return genericplot(celldata, celldata.obsm['Xpca'][:,-3:])

def genericwrongplot(celldata, coords, yhat, labels=None):
    """Plot figure with incorrectly classified points highlighted
    
    This can be used with any 2D or 3D embedding (e.g., PCA or t-SNE). For
    example code, see `pcawrongplot`.

    :param celldata: Rocks object
    :param coords: a (N, 2) or (N, 3) shaped array with coordinates to plot
    :param yhat: (N, 1) shaped array of predicted/guessed y values
    :param labels: (optional) list of axis titles
    """

    y = celldata.obs['y']
    if yhat.shape == (y.shape[0],):
        yhat = yhat.reshape(y.shape)
    assert yhat.shape == y.shape, "yhat must have shape (N, 1)"

    colscal = cl.scales['9']['qual']['Set1']

    # indices (by cluster) where wrong
    wronginds = {}
    for k in range(celldata.K):
        wronginds[k] = np.nonzero((y == k) & np.not_equal(yhat,y))[0]
    
    # indices (by cluster) where correct 
    clustinds = {}
    for k in range(celldata.K):
        clustinds[k] = np.nonzero((y == k) & np.equal(yhat, y))[0]

    def scatter(coords, *args, **kwargs):
        """Run the appropriate scatter function"""
        assert coords.shape[1] in [2,3], "incorrect dimensions for coords"
        if coords.shape[1] == 2:
            return go.Scatter(x=coords[:,0], y=coords[:,1], *args, **kwargs)
        else:
            return go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2],
                    *args, **kwargs)
        
    # Get the points that are wrong
    plotdata = [scatter(
            coords[inds],
            mode='markers',
            marker=dict(
                size=4,
                color=colscal[k % len(colscal)],
                opacity=1),
            name="Cluster {}".format(k),
            hoverinfo="name+text",
            text = ["Predict {}".format(str(a)) for a in yhat[inds]])
            for k, inds in wronginds.items()] + \
        [scatter(
            coords[inds],
            mode='markers',
            marker=dict(
                size=4,
                color=colscal[k % len(colscal)],
                opacity=0.2),
            name="Cluster {}".format(k),
            hoverinfo="name",
            showlegend=False)
            for k, inds in clustinds.items()]
        
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest"
    )
    if labels:
        layout["scene"] = dict(
            xaxis=dict(title=labels[0]),
            yaxis=dict(title=labels[1]),
        )
        if len(labels) == 3:
            layout["scene"]["zaxis"] = dict(title=labels[2])

    return go.Figure(data=plotdata, layout=layout)


def pcawrongplot(celldata, yhat):
    """Generate a 3D PCA figure with incorrectly classified points highlighted

    :param celldata: a Rocks object
    :param yhat: computed (guessed) y vector
    """

    if celldata.Xpca is None or celldata.Xpca.shape[1] < 3:
        print("Need 3 PCs. Calculating now.")
        celldata.pca(3)
    Xpca = celldata.Xpca
    return genericwrongplot(celldata, Xpca[:,-3:], yhat,
            labels=["PC1", "PC2", "PC3"])


def plotgeneheat(celldata, coords, genes):
    """Generate gene heat plot for some embedding of Rocks data

    This generates a figure with multiple dropdown options. The first option is
    "Clust" for a plot similar to `genericplot`, and the remaining dropdown
    options correspond to genes specified in `genes`. When `celldata.genes` is
    defined, these drop downs are labeled with the gene names.

    :param celldata: a Rocks object
    :param coords: (N, 2) or (N, 3) shaped coordinates of the embedded data 
    :param genes: list of gene indices
    """

    def scatter(coords, *args, **kwargs):
        """Run the appropriate scatter function"""
        assert coords.shape[1] in [2,3], "incorrect dimensions for coords"
        if coords.shape[1] == 2:
            return go.Scatter(x=coords[:,0], y=coords[:,1], *args, **kwargs)
        else:
            return go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2],
                    *args, **kwargs)
    numclusts = celldata.K
    clusterindices = celldata.clusterindices
    clustscal = cl.scales['9']['qual']['Set1']
    genescal = np.array(cl.scales['8']['seq']['Blues'])

    plotbyclust = [scatter(
            coords[inds],
            mode='markers',
            marker=dict(
                size=4,
                color=clustscal[k % len(clustscal)], # set color to an array/list
                #                                  of desired values
                ),
            name="Cluster {}".format(k),
            hoverinfo="name"
            )
            for k, inds in clusterindices.items()]
    
    
    geneexpr = celldata.X[:,genes]
    exprnorm = np.round(geneexpr*7/geneexpr.max(axis=0)).astype(int)
    numgenes = geneexpr.shape[1]
    try:
        genenames = celldata.markers_to_genes(genes)
    except ValueError:
        genenames = ["Gene {}".format(genes[i]) for i in range(numgenes)] 
    
    plotbygene = [scatter(
            coords,
            mode='markers',
            marker=dict(
                size=4,
                color=genescal[exprnorm[:,i]],
                ),
            name=genenames[i],
            text=geneexpr[:,i].astype(str),
            hoverinfo="name+text",
            visible=False,
            ) for i, genename in enumerate(genenames)]
    buttons = [dict(label="Clust",
                    method="update",
                     args=[{"visible": [True] * numclusts + [False] * numgenes}, {}]
                    ),]
    for i, genename in enumerate(genenames):
        v = [False] * (numclusts + numgenes)
        v[numclusts + i] = True
        buttons.append(dict(label=genename,
                    method="update",
                    args=[{"visible": v}, {}])
                    )
        
    updatemenus = [dict(buttons=buttons, direction="down",  x=0.1, y=1.1, xanchor="left", yanchor="top", pad={'r':10, 't': 10},)]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        updatemenus=updatemenus,
        annotations=[
            dict(text='View:', x=0.01, y=1.065, xref='paper', yref='paper', align='left', showarrow=False)],
        showlegend=True,
    )

    return go.Figure(data=plotbyclust+plotbygene, layout=layout)
