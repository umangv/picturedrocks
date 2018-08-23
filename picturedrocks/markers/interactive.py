import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import colorlover as cl
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import picturedrocks as pr

_import_errors = None

try:
    import ipywidgets as ipyw
    from IPython.display import display
    from tqdm._tqdm_notebook import tqdm_notebook as tqdm

    tqdm.monitor_interval = False
except ImportError as e:
    _import_errors = e.name


class InteractiveMarkerSelection:
    """Run an interactive marker selection GUI inside a jupyter notebook

    Args
    ----
    adata: anndata.AnnData
        The data to run marker selection on. If you want to restrict to a small
        number of genes, slice your anndata object.
    infoset: picturedrocks.markers.InformationSet
        An InformationSet corresponding to `adata`
    obj: function
        An objective function (see `cife_obj` for an example)
    disp_genes: int
        Number of genes to display as options (by default, number of genes
        plotted on the tSNE plot is `3 * disp_genes`, but can be changed by
        setting the `plot_genes` property after initializing.
    connected: bool
        Parameter to pass to `plotly.offline.init_notebook_mode`. If your
        browser does not have internet access, you should set this to False.

    Warning
    -------
    This class requires modules not explicitly listed as dependencies of
    picturedrocks. Specifically, please ensure that you have `ipywidgets` and
    `tqdm` installed and that you use this class only inside a jupyter notebook.
    """

    def __init__(self, adata, infoset, obj, disp_genes=10, connected=True):
        if _import_errors:
            raise ImportError(f"Unable to import {_import_errors}")
        self.adata = adata
        self.infoset = infoset
        self.obj = obj
        self.S = []
        self.pool = np.arange(adata.n_vars)
        self.scores = np.zeros(len(self.pool))

        init_notebook_mode(connected=connected)

        if not "gene_tsne" in self.adata.varm_keys():
            print("Running tSNE on genes...")
            p = PCA(n_components=30)
            t = TSNE()
            self.adata.varm["gene_pca"] = p.fit_transform(self.adata.X.T)
            self.adata.varm["gene_tsne"] = t.fit_transform(self.adata.varm["gene_pca"])

        if not "X_tsne" in self.adata.obsm_keys():
            print("Running tSNE on cells...")
            p = PCA(n_components=30)
            t = TSNE()
            self.adata.obsm["X_pca"] = p.fit_transform(self.adata.X)
            self.adata.obsm["X_tsne"] = t.fit_transform(self.adata.obsm["X_pca"])

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

    def compute_redraw(self):
        """Recompute scores and redraw jupyter widgets"""
        H = self.infoset.H
        self.out_next.children = []
        self.out_cur.children = []

        self.out_plot.clear_output()

        with self.out_plot:
            self.scores = np.array([self.obj(H, i, self.S) for i in tqdm(self.pool)])
        baseline_score = self.obj(H, self.infoset.baseline_index, self.S)
        self.scores -= baseline_score

        top_gene_inds = np.argsort(self.scores)[::-1]

        self.out_next.children = (
            [ipyw.Label("Candidate Next Gene")]
            + [
                self._next_gene_row(gene_ind, self.scores[gene_ind])
                for gene_ind in top_gene_inds[: self.disp_genes]
            ]
            + [self._other_next_gene()]
        )

        self.out_cur.children = [ipyw.Label("Currently selected genes")] + [
            self._cur_gene_row(gene_ind) for gene_ind in self.S
        ]

        ninfscores = self.scores == float("-inf")
        scaled_scores = self.scores - self.scores[~ninfscores].min()
        scaled_scores[scaled_scores < 0] = 0
        scaled_scores = scaled_scores / scaled_scores.max()

        top_genes_plot = top_gene_inds[: self.plot_genes]
        self.out_plot.clear_output()
        with self.out_plot:
            iplot(
                pr.plot.plotgeneheat(
                    self.adata,
                    self.adata.obsm["X_tsne"],
                    top_gene_inds[: self.disp_genes].tolist() + self.S,
                )
            )
            iplot(
                go.Figure(
                    data=[
                        go.Scatter(
                            x=self.adata.varm["gene_tsne"][top_genes_plot, 0],
                            y=self.adata.varm["gene_tsne"][top_genes_plot, 1],
                            text=self.adata.var_names.values.astype(str)[
                                top_genes_plot
                            ],
                            mode="markers",
                            hoverinfo="text",
                            marker=dict(
                                size=(scaled_scores[top_genes_plot] * 16).astype(int),
                                color=np.array(cl.scales["9"]["seq"]["Blues"])[
                                    (3 + 5 * scaled_scores[top_genes_plot].astype(int))
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
            self.S.append(gene_ind)
            self.compute_redraw()

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
            self.S.remove(gene_ind)
            self.compute_redraw()

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
                label.value = "(score: {:0.4f})".format(self.scores[gene_ind])
                but.disabled = False
            except KeyError:
                gene_ind = -1
                label.value = "(?)"
                but.disabled = True

        textbox.observe(name_updated, names="value")

        def add_other_gene(b):
            if gene_ind >= 0:
                self.S.append(gene_ind)
                self.compute_redraw()

        but.on_click(add_other_gene)
        return ipyw.HBox([but, textbox, label])

    def show(self):
        """Display the jupyter widgets"""
        display(self.out)
        self.compute_redraw()


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
