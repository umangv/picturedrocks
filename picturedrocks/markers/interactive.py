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

from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import colorlover as cl
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objects as go
import picturedrocks as pr

_import_errors = None

try:
    import ipywidgets as ipyw
    from IPython.display import display
except ImportError as e:
    _import_errors = e.name


class InteractiveMarkerSelection:
    def __init__(
        self, adata, feature_selection, visuals=None, disp_genes=10, connected=True
    ):
        """Run an interactive marker selection GUI inside a jupyter notebook

        Args
        ----
        adata: anndata.AnnData
            The data to run marker selection on. If you want to restrict to a small
            number of genes, slice your anndata object.
        feature_selection: picturedrocks.markers.mutualinformation.iterative.IterativeFeatureSelection
            An instance of a interative feature selection algorithm class
            that corresponds to `adata` (i.e., the column indices in
            `feature_selection` should correspond to the column indices in
            `adata`)
        visuals: list
            List of visualizations to display. These can either be shorthands
            for built-in visualizations (currently "tsne", "umap", and
            "violin"), or an instance of InteractiveVisualization (see
            GeneHeatmap and ViolinPlot for example implementations).
        disp_genes: int
            Number of genes to display as options (by default, number of genes
            plotted on the tSNE plot is `3 * disp_genes`, but can be changed by
            setting the `plot_genes` property after initializing.
        connected: bool
            Parameter to pass to ``plotly.offline.init_notebook_mode``. If your
            browser does not have internet access, you should set this to False.

        Warning
        -------
        This class requires modules not explicitly listed as dependencies of
        picturedrocks. Specifically, please ensure that you have `ipywidgets`
        installed and that you use this class only inside a jupyter notebook.
        """

        if _import_errors:
            raise ImportError(f"Unable to import {_import_errors}")
        self.adata = adata
        self.featsel = feature_selection

        init_notebook_mode(connected=connected)

        self.disp_genes = disp_genes
        self.top_genes = []

        self.out_next = ipyw.VBox()
        self.out_cur = ipyw.VBox()

        def _get_visual_object(obj):
            if type(obj) == str:
                if obj == "tsne":
                    return GeneHeatmap("tsne")
                elif obj == "umap":
                    return GeneHeatmap("umap")
                elif obj == "violin":
                    return ViolinPlot()
                else:
                    raise ValueError(f"Invalid visualization shorthand: {obj}")
            elif isinstance(obj, InteractiveVisualization):
                return obj
            else:
                raise ValueError(
                    "Visual must be a shorthand or an instance of InteractiveVisualization"
                )

        if visuals is None:
            self.visuals = [GeneHeatmap()]
        else:
            self.visuals = [_get_visual_object(vis) for vis in visuals]

        def _tab_changed(change):
            if change["new"] is not None:
                self._draw_visual(change["new"])

        self.out_visuals = [
            ipyw.Output(layout=ipyw.Layout(width="100%")) for _ in self.visuals
        ]
        self.visuals_drawn = [False] * len(self.visuals)
        self.out_plot = ipyw.Tab(children=self.out_visuals)
        self.out_plot.observe(_tab_changed, "selected_index")
        for i, (out, vis) in enumerate(zip(self.out_visuals, self.visuals)):
            vis.prepare(self.adata, out)
            self.out_plot.set_title(i, vis.title)

        self.out = ipyw.Output()

        with self.out:
            display(
                ipyw.VBox([ipyw.HBox([self.out_next, self.out_cur]), self.out_plot])
            )

    def show_loading(self):
        self.out_next.children = [ipyw.Label("Loading...")]
        self.out_cur.children = []

    def _draw_visual(self, visual_ind):
        """Lazily draw visualization"""
        if not self.visuals_drawn[visual_ind]:
            self.visuals[visual_ind].redraw(self.top_genes, self.featsel.S)
            self.visuals_drawn[visual_ind] = True

    def redraw(self):
        """Redraw jupyter widgets after a change
        
        This is called internally and there should usually be no need for the
        user to explicitly call this method."""
        self.out_next.children = []
        self.out_cur.children = []

        top_gene_inds = np.argsort(self.featsel.score)[::-1]
        self.top_genes = top_gene_inds[: self.disp_genes].tolist()

        ninfscores = self.featsel.score == float("-inf")
        scaled_scores = self.featsel.score - self.featsel.score[~ninfscores].min()
        scaled_scores[scaled_scores < 0] = 0
        scaled_scores = scaled_scores / scaled_scores.max()

        self.out_next.children = (
            [ipyw.Label("Candidate Next Gene")]
            + [
                self._next_gene_row(gene_ind, scaled_scores[gene_ind])
                for gene_ind in self.top_genes
            ]
            + [self._other_next_gene()]
        )

        self.out_cur.children = [ipyw.Label("Currently selected genes")] + [
            self._cur_gene_row(gene_ind) for gene_ind in self.featsel.S
        ]

        self.visuals_drawn = [False] * len(self.visuals)
        for out in self.out_visuals:
            out.clear_output()
        self._draw_visual(self.out_plot.selected_index)

    def _next_gene_row(self, gene_ind, score):
        but = ipyw.Button(
            icon="plus",
            tooltip="Add {}".format(self.adata.var_names[gene_ind]),
            layout=dict(width="40px"),
        )

        def add_cur_gene(b):
            self.show_loading()
            self.featsel.add(gene_ind)
            self.redraw()

        but.on_click(add_cur_gene)
        return ipyw.HBox(
            [
                but,
                ipyw.Label(
                    "{} (score: {:0.4f})".format(self.adata.var_names[gene_ind], score),
                    layout=dict(width="250px"),
                ),
            ]
        )

    def _cur_gene_row(self, gene_ind):
        def del_cur_gene(b):
            self.show_loading()
            self.featsel.remove(gene_ind)
            self.redraw()

        pop_but = ipyw.Button(
            icon="minus-circle",
            tooltip="Remove {}".format(self.adata.var_names[gene_ind]),
            layout=dict(width="40px"),
        )
        pop_but.on_click(del_cur_gene)
        return ipyw.HBox(
            [
                ipyw.Label(self.adata.var_names[gene_ind], layout=dict(width="250px")),
                pop_but,
            ]
        )

    def _other_next_gene(self):
        but = ipyw.Button(
            icon="plus", tooltip="Add", disabled=True, layout=dict(width="40px")
        )
        label = ipyw.Label("(?)")
        gene_ind = -1
        textbox = ipyw.Text(layout=dict(width="150px"), placeholder="Gene Name")

        def name_updated(change):
            nonlocal gene_ind
            try:
                gene_ind = self.adata.var_names.get_loc(textbox.value)
                label.value = "(score: {:0.4f})".format(self.featsel.score[gene_ind])
                but.disabled = False
            except KeyError:
                gene_ind = -1
                label.value = "(?)"
                but.disabled = True

        textbox.observe(name_updated, names="value")

        def add_other_gene(b):
            if gene_ind >= 0:
                self.show_loading()
                self.featsel.add(gene_ind)
                self.redraw()

        but.on_click(add_other_gene)
        return ipyw.HBox([but, textbox, label])

    def show(self):
        """Display the jupyter widgets"""
        display(self.out)
        self.redraw()


class InteractiveVisualization(ABC):
    def __init__(self):
        """Abstract base class for interactive visualizations

        Extend this class and pass an instance of it to
        InteractiveMarkerSelection to use your own visualization. It is
        recommended that you begin your implementation of ``__init__`` with::

            super().__init__()

        You are welcome to add parameters specific to your visualization in the
        ``__init__`` method.
        """
        self.adata = None
        self.out = None

    def prepare(self, adata, out):
        """Prepare for visualization

        This method is called when InteractiveMarkerSelection is initialized.
        It is recommended that you begin your implementation with ::

            super().prepare(adata, out)

        This stores adata and out in ``self.adata`` and ``self.out``
        respectively.
        """
        self.adata = adata
        self.out = out

    @property
    def title(self):
        """Title of the visualization
        
        This should be a Python property, using the ``@property``
        decorator.
        """
        return "Untitled"

    @abstractmethod
    def redraw(self, next_gene_inds, cur_gene_inds):
        """Draw the visualization

        You must implement this method. To display the plots in the
        appropriate widget, use::

            with self.out:
                fig.show() # or your plotting library's equivalent
        """
        pass


class GeneHeatmap(InteractiveVisualization):
    def __init__(self, dim_red="tsne", n_pcs=30):
        """GeneHeatmap for Interactive Marker Selection

        Args
        ----
        dim_red: str
            Dimensionality reduction algorithm to use. Currently available options are "umap" and "tsne"
        
        n_pcs: int
            The number of principal components to map to before running dimensionality reduction
        """
        super().__init__()
        assert dim_red in ["tsne", "umap"]
        self._dim_red = dim_red
        self.n_pcs = n_pcs

    @property
    def title(self):
        return "{} Heatmap".format({"tsne": "t-SNE", "umap": "UMAP"}[self._dim_red])

    def prepare(self, adata, out):
        super().prepare(adata, out)
        if ("X_" + self._dim_red) not in self.adata.obsm_keys():
            print(f"Running {self._dim_red} on cells...")
            p = PCA(n_components=self.n_pcs)
            if self._dim_red != "pca":
                dr = {"tsne": TSNE, "umap": UMAP}[self._dim_red]()
                self.adata.obsm["X_pca"] = p.fit_transform(self.adata.X)
                self.adata.obsm["X_" + self._dim_red] = dr.fit_transform(
                    self.adata.obsm["X_pca"]
                )

    def redraw(self, next_gene_inds, cur_gene_inds):
        with self.out:
            fig = pr.plot.plotgeneheat(
                self.adata,
                self.adata.obsm["X_" + self._dim_red],
                next_gene_inds + cur_gene_inds,
            )
            iplot(fig)


class ViolinPlot(InteractiveVisualization):
    def __init__(self):
        """Violin Plots for each class label"""
        super().__init__()

    @property
    def title(self):
        return "Violin Plots"

    def prepare(self, adata, out):
        super().prepare(adata, out)

    def redraw(self, next_gene_inds, cur_gene_inds):
        fig = go.Figure()
        gene_inds = next_gene_inds + cur_gene_inds
        buttons = []
        for i, gind in enumerate(gene_inds):
            visible = [False] * len(gene_inds)
            visible[i] = True
            buttons.append(
                dict(
                    label=self.adata.var_names[gind],
                    method="update",
                    args=[{"visible": visible}],
                )
            )
            fig.add_trace(
                go.Violin(
                    x=self.adata.obs["clust"],
                    y=self.adata.X[:, gind],
                    points="all",
                    visible=(i == 0),
                )
            )
        updatemenus = [
            dict(
                buttons=buttons,
                direction="down",
                x=0,
                y=1.1,
                xanchor="left",
                yanchor="top",
                pad={"r": 0, "t": 0},
            )
        ]

        fig.update_layout(updatemenus=updatemenus)
        with self.out:
            fig.show()
