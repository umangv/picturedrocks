import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import colorlover as cl
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import picturedrocks as pr

_import_errors = None

try:
    import ipywidgets as ipyw
    from IPython.display import display
except ImportError as e:
    _import_errors = e.name


class InteractiveMarkerSelection:
    def __init__(
        self,
        adata,
        feature_selection,
        disp_genes=10,
        connected=True,
        show_cells=True,
        show_genes=True,
        dim_red="tsne",
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
        disp_genes: int
            Number of genes to display as options (by default, number of genes
            plotted on the tSNE plot is `3 * disp_genes`, but can be changed by
            setting the `plot_genes` property after initializing.
        connected: bool
            Parameter to pass to `plotly.offline.init_notebook_mode`. If your
            browser does not have internet access, you should set this to False.
        show_cells: bool
            Determines whether to display a tSNE plot of the cells with a
            drop-down menu to look at gene expression levels for candidate
            genes.
        show_genes: bool
            Determines whether to display a tSNE plot of genes to visualize
            gene similarity
        dim_red: {"tsne", "umap"}
            Dimensionality reduction algorithm

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

        self.show_genes = show_genes
        self.show_cells = show_cells
        self.dim_red = dim_red

        init_notebook_mode(connected=connected)

        if show_genes or show_cells:
            assert dim_red in [
                "tsne",
                "umap",
            ], "Invalid dimensionality Reduction Algorithm"
            dim_red_cls = {"tsne": TSNE, "umap": UMAP}[dim_red]

        if show_genes and ("gene_" + dim_red) not in self.adata.varm_keys():
            print(f"Running {dim_red} on genes...")
            p = PCA(n_components=30)
            dr = dim_red_cls()
            self.adata.varm["gene_pca"] = p.fit_transform(self.adata.X.T)
            self.adata.varm["gene_" + dim_red] = dr.fit_transform(
                self.adata.varm["gene_pca"]
            )

        if show_cells and ("X_" + dim_red) not in self.adata.obsm_keys():
            print(f"Running {dim_red} on cells...")
            p = PCA(n_components=30)
            dr = dim_red_cls()
            self.adata.obsm["X_pca"] = p.fit_transform(self.adata.X)
            self.adata.obsm["X_" + dim_red] = dr.fit_transform(self.adata.obsm["X_pca"])

        self.disp_genes = disp_genes
        self.plot_genes = disp_genes * 3

        self.out_next = ipyw.VBox()
        self.out_cur = ipyw.VBox()
        self.out_plot = ipyw.Output()
        self.out = ipyw.Output()

        with self.out:
            display(
                ipyw.VBox([ipyw.HBox([self.out_next, self.out_cur]), self.out_plot])
            )

    def show_loading(self):
        self.out_next.children = [ipyw.Label("Loading...")]
        self.out_cur.children = []

    def redraw(self):
        """Redraw jupyter widgets"""
        self.out_next.children = []
        self.out_cur.children = []

        self.out_plot.clear_output()

        top_gene_inds = np.argsort(self.featsel.score)[::-1]

        self.out_next.children = (
            [ipyw.Label("Candidate Next Gene")]
            + [
                self._next_gene_row(gene_ind, self.featsel.score[gene_ind])
                for gene_ind in top_gene_inds[: self.disp_genes]
            ]
            + [self._other_next_gene()]
        )

        self.out_cur.children = [ipyw.Label("Currently selected genes")] + [
            self._cur_gene_row(gene_ind) for gene_ind in self.featsel.S
        ]

        ninfscores = self.featsel.score == float("-inf")
        scaled_scores = self.featsel.score - self.featsel.score[~ninfscores].min()
        scaled_scores[scaled_scores < 0] = 0
        scaled_scores = scaled_scores / scaled_scores.max()

        top_genes_plot = top_gene_inds[: self.plot_genes]
        self.out_plot.clear_output()
        with self.out_plot:
            if self.show_cells:
                iplot(
                    pr.plot.plotgeneheat(
                        self.adata,
                        self.adata.obsm["X_" + self.dim_red],
                        top_gene_inds[: self.disp_genes].tolist() + self.featsel.S,
                    )
                )
            if self.show_genes:
                iplot(
                    go.Figure(
                        data=[
                            go.Scatter(
                                x=self.adata.varm["gene_" + self.dim_red][
                                    top_genes_plot, 0
                                ],
                                y=self.adata.varm["gene_" + self.dim_red][
                                    top_genes_plot, 1
                                ],
                                text=self.adata.var_names.values.astype(str)[
                                    top_genes_plot
                                ],
                                mode="markers",
                                hoverinfo="text",
                                marker=dict(
                                    size=(scaled_scores[top_genes_plot] * 16).astype(
                                        int
                                    ),
                                    color=np.array(cl.scales["9"]["seq"]["Blues"])[
                                        (
                                            3
                                            + 5
                                            * scaled_scores[top_genes_plot].astype(int)
                                        )
                                    ],
                                    line=go.scatter.marker.Line(color="black", width=1),
                                ),
                            )
                        ],
                        layout=go.Layout(hovermode="closest"),
                    )
                )

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


def cife_obj(H, i, S):
    """The CIFE objective function for feature selection
    
    Args
    ----
    H: function
        an entropy function, typically the bound method H on an instance of
        InformationSet. For example, if `infoset` is of type
        `picturedrocks.markers.InformationSet`, then pass `infoset.H`
    i: int
        index of candidate gene
    S: list
        list of features already selected
    Returns
    -------
    float
        the candidate feature's score relative to the selected gene set `S`
    """
    Sset = set(S)
    m = len(S)
    if i in Sset:
        return float("-inf")
    curobj = (1 - m) * (H((i,)) - H((-1, i)))
    for x in S:
        curobj += H((x, i)) - H((-1, x, i))
    return curobj
