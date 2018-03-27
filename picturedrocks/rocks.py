# Copyright © 2017 Anna Gilbert, Alexander Vargo, Umang Varma
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
import scipy.spatial.distance as scipydist
from scipy.sparse.linalg import svds

# Currently need this for 1CS method (along with numpy)
import cvxpy as cvx

# Helper functions for the 1 bit compressed sensing method

#### Correlations, as in SPA Stanardize
#
# This computes a vector. The $i$-th entry is the correlation (as defined in
# section 3.2.3 of Genzel's thesis) between feature vector corresponding to gene
# $i$ (vec) and the cluster data corresponding to the input clusters (mat). 

# compute corrlation of columns of mat with vec
def corrVec(vec, mat):
    matBar = mat.mean(axis=0)
    vecBar = vec.mean()
    
    r_num = np.sum( (mat-matBar)*(vec[:,None]-vecBar), axis=0) *1.0
    r_den = vec.shape[0]*np.std(vec)*np.std(mat,axis=0)

    if (len(list(r_den[r_den == 0])) != 0): 
        r_den[r_den == 0] = r_den[r_den == 0] + 0.000000001
            
    return r_num/(r_den)

# Generate a list of values of epsilon to test.
def genEpsList(nsteps=10, minEps=10.0**(-5), maxEps=10.0**(-3)):
    stepSize = (maxEps - minEps)/(nsteps*1.0)
        
    epsList = [maxEps - mult* stepSize for mult in range(nsteps+1)]
    return epsList


def pca(Xin, dim=3):
    Xbar = Xin.mean(axis=0)
    Xcent = Xin - Xbar
    print("Computing Corvariance Matrix")
    Sigma = np.cov(Xcent.T)
    print("Performing SVDs")
    pcs = svds(Sigma, dim, return_singular_vectors="u")
    Xpca = Xcent.dot(pcs[0])
    return (Xcent, Sigma, pcs, Xpca)


class Rocks:
    """PicturedRocks Single-Cell RNA-seq Analysis Tool

     :param X: the gene expression matrix. We expect a numpy array of shape (N, P)
         containing data for N cells and P genes (note rows are cells and
         columns are genes.
     :param y: cluster labels for N cells. We expect a numpy array of shape (N,
         1) or (N,) taking values 0, 1, 2, ..., K - 1. 
     :param genes: (optional) names of genes. We expect an array of P strings,
         containing the names of various genes
     :param verbose: verbosity level for debugging; defaults to 0.

     For example, if you import your data using a `pandas` dataframe with P
     genes (columns) and N rows (cells) and a "Cluster" column as the last
     column, you can create a Rocks object as follows::

         X = df.as_matrix()[:,:-1]
         y = df.as_matrix()[:,-1]
         genes = dataframe.columns
         rocks = Rocks(X, y, genes)
     """
    def __init__(self, X, y, genes=None, verbose=0):
        self.verbose = verbose
        # self.X is the expression data
        self.X = X
        # self.y is the cluster assignment
        self.y = y
        # self.N is the number of cells
        # self.P is the number of genes
        self.N, self.P = X.shape

        # self.genes contains the names of the genes
        self.genes = genes
        assert (genes is None or len(genes) == self.P), \
                "genes must be an array of length P or None"

        # Some extra elements that will be needed for OvA
        # they will be changed in the methods below
        self.cs_currY = self.y
        self.cs_currX = self.X

        
        self.Xcent, self.pcs, self.Xpca = (None, None, None)
        if y.shape == (self.N,):
            self.y = y.reshape((self.N, 1))
        assert self.y.shape == (self.N, 1), \
                "y should be a matrix of shape (N,1)"
        
        self.K = self.y.max() + 1
        # self.K is the number of clusters
        assert np.array_equal(np.unique(self.y), range(self.K)), \
                "Cluster labels should be 0, 1, 2, ..., K -1"
        
        self.clusterindices = {}
        for k in range(self.K):
            self.clusterindices[k] = np.nonzero(self.y == k)[0]
        nk = np.array([len(self.clusterindices[k]) for k in range(self.K)])
    
    def _debug(self, level, message):
        if self.verbose >= level:
            print(message)

    # markers can be a list or 1-dim np array
    def markers_to_genes(self, markers):
        """Convert gene indices to gene names

        :param markers: a list of marker indices (numbers between 0 and P)
        :returns: a list of gene names

        .. note::
            to use this method, Rocks.genes must have been specified, either at
            initialization or manually via ``object.genes = ...``
        """

        try:
            return [self.genes[a] for a in markers]
        except TypeError:
            raise ValueError("Gene names not specified. Set using object.genes")

    
    def normalize(self, totalexpr="median", log=True):
        """Normalize data

        :param totalexpr: the total expression to normalize every cell to. For
            example, if `totalexpr =  1000`, each cell is scaled so that its
            expression levels add to 1000. If `totalexpr = "median"` (default
            behavior), we set the total expression to the median total
            expression across all cells.
        :param log: whether to log-transform the data (we always add 1 before
            taking logs).
        """
        cellsize = self.X.sum(axis=1).reshape((self.N,1))
        targetsize = np.median(cellsize) if totalexpr == "median" else totalexpr

        # avoid zero_divide issues
        cellsize[cellsize == 0] = 1

        self.X = (targetsize*self.X)/cellsize
        if log:
            self.X = np.log(self.X +1)
        self.Xcent, self.pcs, self.Xpca = (None, None, None)
    
    def pca(self, dims=3):
        """Perform PCA on the data

        :param dims: number of dimensions to consider in PCA
        """
        self.Xcent, Sigma, self.pcs, self.Xpca = pca(self.X, dims)
        self.totalvariation = np.trace(Sigma)
        # Sigma is too big to be worth storing in memory

    
    def markers_mutualinfo(self, n, pool = None):        
        """Compute markers using mutual information

        :param n: the number of markers to select
        :param pool: (optional) pool of genes to restrict marker selection
            search to
        """
        import datetime
        X = np.log2(self.X+1).round().astype(int)
        if pool is None:
            pool = range(self.P)
        
        maxentry = max(X.max(), self.y.max())
        base = 10**int(round(np.log10(maxentry) + 1))

        def I(cx, cy):
            xy = np.array([cx, cy]).T
            xyconcat = xy.dot(np.array([base, 1]))
            values, counts = np.unique(xyconcat, return_counts=True)
            valcount = zip(list(values), list(counts))
            pxy = np.zeros((maxentry + 1, maxentry + 1))
            px = np.zeros(maxentry + 1)
            py = np.zeros(maxentry + 1)
            for value, count in valcount:
                xval = value // base
                yval = value % base
                pxy[xval, yval] += count
                px[xval] += count
                py[yval] += count
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (pxy/np.expand_dims(px, axis=1))/np.expand_dims(py, axis=0) * n
                r = np.sum(np.nan_to_num((pxy/n) * np.log(s)))
            return r
        
        self._debug(1, "Computing I(x_i, y) values...")
        yflat = self.y.flatten()
        Ixiy = np.zeros(self.P)
        for i in pool:
            Ixiy[i] = I(X[:,i], yflat)
        
        Ixixj = np.zeros((self.P,self.P))
        IxixjChecked = np.zeros((self.P,self.P)).astype(bool)

        def getIxixj(i,j):
            if not IxixjChecked[i,j]:
                Ixixj[i, j] = I(X[:, i], X[:,j])
                Ixixj[j, i] = Ixixj[i, j]
                IxixjChecked[i,j] = IxixjChecked[j,i] = True
            return Ixixj[i, j]

        start = datetime.datetime.now()
        S = []
        Phi = 0
        self._debug(1, "Selecting candidate features...")
        for m in range(n):
            self._debug(1, "m = {}".format(m))
            maxDeltaPhi = float("-inf") # max increase in Phi
            argmaxDeltaPhi = -1 # index  i that corresponds to the value above
            for i in pool:
                if i in S:
                    continue
                DeltaPhi = Ixiy[i]
                if m > 0:
                    DeltaPhi -= (1.0/m) * np.sum([getIxixj(i, j) for j in S])
                if DeltaPhi > maxDeltaPhi:
                    maxDeltaPhi = DeltaPhi
                    argmaxDeltaPhi = i
                    self._debug(2,
                            "Feature provisionally selected m = {}, i = {}".\
                                    format(m, i))
            S.append(argmaxDeltaPhi)
            self._debug(1, "Features: {}".format(S))
        end = datetime.datetime.now()
        timedelta = end-start
        self._debug(1,
                "It took {:.2f} minutes to find the {} features via mRMR."\
                        .format(timedelta.total_seconds()/60, n))
        return S


    # THE 1 BIT COMPRESSED SENSING METHOD (1CS, GENZEL)
    #
    # The methods below implement only one-vs-rest (ovr) multiclass
    # classification.  All-vs-all will be added soon
    #
    # TODO: Some functions below might be repeats of earlier functions

    # Get the indices of cells in cluster clustind
    def clust2vec(self, clustind=1):
        returnvec = -1.0*np.ones(self.cs_currY.shape[0])
        returnvec[[i for i,x in enumerate(self.cs_currY) if x==clustind - 1]] \
                = 1.0
        return returnvec

    # Make the appropriate vector for 1CS methods
    # the vector of coefficients used in the optimization problem
    def coeffs(self, clustind=1):
        return np.sum( self.clust2vec(clustind)[0:,np.newaxis]*self.cs_currX,
                axis=0 )

    # Find the transform needed for scaling the data
    def findXform(self, alpha=0.3, c=2.0, lambFac=1.0, clustOne=1, clustTwo=None):
        # print("Finding transform with alpha = {}, c = {}, lambFac = {}, cluster
        # = {}".format(alpha, c, lambFac, clustOne), flush=True)
       
        # restrict the data if we are looking at two clusters
        if (clustTwo):
            self.cs_currY = np.concatenate( [self.y[self.y ==clustOne],
                self.y[self.y==clustTwo]] )
            self.cs_currX = np.concatenate( [self.X[
                np.squeeze(np.asarray(self.y==clustOne)) ],
                self.X[ np.squeeze(np.asarray(self.y==clustTwo)) ]] )
        
        # find vector of correlations
        rho = corrVec( self.clust2vec(clustOne), self.cs_currX )
        # print("Correlations: {}".format(rho[0:10]), flush=True)
        
        # find scaling vector
        sigma = np.std(self.cs_currX, axis=0)
        sigma[sigma==0] = 0.0000001
        alphaFac = alpha**(c * ( 1-np.abs(rho)))
        scaleFac = lambFac*alphaFac + (1 - alphaFac)/sigma
        
        # find center
        xbar = self.cs_currX.mean(axis=0)
        
        # TODO: do we actually need to return something here?
        return xbar, scaleFac

    # Implement the optimization using the cvxpy methods
    #
    # When we call this, we need self.cs_currX to be exactly the data we want to
    # use in the optimization That is, it should be standardized and transformed
    # in every way that we want.  Also, lamb is the overal lambda constraint on
    # the 1-norm
    def findW(self, clustInd, lamb, tol=10.0**(-3)):
        consts = self.coeffs(clustInd)
        
        w = cvx.Variable(self.P) # self.P is the number of genes
        
        constraints = [ cvx.pnorm(w,1) <= lamb, cvx.norm(w,2) <= 1.0 ]
        objective = cvx.Maximize( cvx.sum_entries(
            cvx.mul_elemwise(consts[0:,np.newaxis],w) ))
        prob= cvx.Problem(objective,constraints)

        prob.solve(solver="SCS")
        
        # consistency checks
        if (np.sum(w.value) - lamb > tol):
            print("Warning for cluster {} - 1 norm condition failed: {}"\
                    .format(clustInd, np.sum(w.value)),flush=True)
        
        tmp = 0
        for i in range(self.cs_currX.shape[1]):
            tmp += w.value[i,0]**2
            
        if (tmp - 1 > tol):
            print("Warning for cluster {} - 2 norm condition failed: {}"\
                    .format(clustInd, tmp), flush=True)
        
        
        return w.value

    # The functions below are used when we transform the data back to a
    # consistent space and then find margins.
    #
    # Since we transform the data back to a consistent space, we don't need to
    # worry about keeping the transformed data around in order to find the
    # margins, so we can do everything on the full list of dubs

    # Input: one dubs vector (as a numGenes x 1 matrix) desired data in
    # consistent space (numCells x numGenes matrix)

    # TODO: Be careful which data you are using here.  Is self.cs_currX correct,
    # or do we need to input the data that we are using?  Answer: for now, we
    # are fine.  We transform the dubs back into the standard way of looking at
    # things.  So we just use self.cs_currX (which should be a centered version
    # of self.X for OVR),
    def dubs2margin(self, eps, dubs):
        
        epsDubs = np.copy(dubs)
        epsDubs[abs(epsDubs) < eps] = 0
        
        return np.squeeze(np.asarray(np.dot(self.cs_currX, epsDubs)))

    # dubs should be a list of (numGenes x 1) matrices, each matrix specifies an
    # optimal hyperplane
    #
    # eps is the cuttoff - we make all entries of the obtained dubs vector 0 if
    # they are smaller than eps
    #
    # desired data in consistent space (numCells x numGenes matrix)
    #
    # Output: all margins for a specific value of epsilon
    def genMarginsDubs(self, eps, dubs):
        
        margins = [self.dubs2margin(eps, w) for w in dubs]
        return margins

    # TODO: this is probably not the best way to accomplish this, since we are
    # really using a different clustering method to determine the accuracy of
    # the selected set of markers.  It should probably be removed.
    #
    # This checks for how many we get correct based on  the given cluster data
    # margins are the output from the above - all margins for a specific value
    # of epsilon.
    def numCorrect(self, margins):

        final = np.argmax(np.array(margins), axis=0)
        # how many classified correctly
        return self.cs_currY[final[:,np.newaxis]==self.cs_currY].shape[0]

    # Run the method in a simple way
    # 
    # The following method runs the ovr method for simple cases.
    def runOvr(self, currLamb, lambFac, alpha, epsList=genEpsList()):

        numClusts = self.K
        
        self._debug(1, "Working on lamb = {}".format(currLamb))
        
        dubs = []
        margins = []

        for clust in range(1,numClusts+1):

            # Don't think that this is needed, but just being safe
            self.cs_currX = self.X
            cent,scale = self.findXform(alpha=alpha, c=2.0, lambFac=lambFac,
                    clustOne=clust)
            # print("Transform center and scale: {}, {}".format(cent, scale),
            # flush=True)

            # transform the data into the proper space for marker selection and
            # find the markers since this is OvR, we will currently have that
            # self.cs_currX = self.X Thus, there is some sketchy use of
            # self.cs_currX here we reset self.cs_currX = self.X at the end to
            # get rid of the memory.
            self.cs_currX = (self.X - cent)*scale
            currDub = self.findW(clust, currLamb)

            # rescale the weight vector and save it
            dubs.append(np.squeeze(np.asarray(currDub))*scale)

            # reset currX to free up memory
            self.cs_currX = self.X
            
        
        # calculate all margins and how many cells we classify correctly for
        # each value in the epslist.  in OVR, our center stays the same (since
        # we are always looking at all of the data).
        # Note that we are putting the ORIGINAL DATA back into the
        # classification method
        # TODO: Should probably remove this step.  Just pick and eps and stick
        # with it.
        cent = self.X.mean(axis=0)
        self.cs_currX = self.X - cent
        margins = [ self.genMarginsDubs(eps, dubs) for eps in epsList ]

        ###
        #print("Margins:", flush=True)
        #for marg in margins:
        #        print("{}".format(marg[0:10]))

        final = [ self.numCorrect(marg) for marg in margins ]
        epsInd = np.argmax(np.array(final))

        bestEps = epsList[epsInd]
        currCorrect = final[epsInd]
        # assume monotonic - we always do better as we decrease epsilon
        # This is not really the case.
        change = (epsInd == len(final)-1)
        
        self._debug(1, "Summary: Number of correctly classified cells for"
                " all values of epsilon")
        self._debug(1, "Summary: {}".format(final))
        self._debug(1, "Summary: bestEps = {}, correctly classified cells = {}"\
                .format(bestEps, currCorrect))
        
        if (change):
            self._debug(1, "Warning: Optimal eps had change=True")
                
        return [dubs, np.array(margins[epsInd]), epsInd]

    
    # run the method and save a list of markers 
    # default behavior: alpha = 0 (no transform used on the data)
    # if you input a value for alpha, lambFac will default to 0
    def simpleCS(self, currLamb, writeOut = True, alpha = 0.0, lambFac = 0.0,
        epsList=genEpsList()):

        dubs,margins,epsInd = self.runOvr(currLamb=currLamb, lambFac=lambFac,
            alpha=alpha, epsList=epsList)

        print("Found markers", flush=True)

        # the actual classification - note that we don't use this for anything
        # right now
        classes = np.argmax(margins, axis=0)
        currCorrect = self.y[ classes[:,np.newaxis]==self.y ].shape[0]

        # find the support genes by truncating according to the best value of
        # epsilion
        bestEps = epsList[epsInd]
        print("Testing: bestEps = {} with currCorrect = {}".format(bestEps,
            currCorrect), flush=True)
        for w in dubs:
            w[abs(w) < bestEps] = 0

        support_genes = [np.nonzero(w)[0] for w in dubs]

        # write the support genes to a data file
        if (writeOut):
            geneFile = "ovrGenes-lamb{}-lFac{}-alpha{}.dat".format(currLamb,
                    lambFac, alpha)
            gFile = open(geneFile, 'w')
            for genes in support_genes:
                for gene in genes:
                    gFile.write("{} ".format(gene))
                gFile.write("\n")
        
        # Return the classifcation and indices of the genes used in the
        # hyperplanes
        return [classes, support_genes]
    

    # Makes a list out of a collection of support genes
    def support2list(self, sgenes):
        import itertools
        flattened = itertools.chain.from_iterable(sgenes)
        return list(set(flattened))

    def markers_CS(self, currLamb, writeOut=False, alpha = 0.0, lambFac = 0.0,
            epsList=genEpsList()):
        
        [classes, support_genes] =  self.simpleCS(currLamb, writeOut,
                alpha=alpha, lambFac=lambFac, epsList=epsList)
        return self.support2list(support_genes)

def genericplot(celldata, coords):
    """Generate a figure for some embedding of Rocks data

    :param celldata: a Rocks object
    :param coords: (N, 2) or (N, 3) shaped coordinates of the embedded data 
    """
    import colorlover as cl
    import plotly.graph_objs as go

    def scatter(coords, *args, **kwargs):
        """Run the appropriate scatter function"""
        assert coords.shape[1] in [2,3], "incorrect dimensions for coords"
        if coords.shape[1] == 2:
            return go.Scatter(x=coords[:,0], y=coords[:,1], *args, **kwargs)
        else:
            return go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2],
                    *args, **kwargs)
    clusterindices = celldata.clusterindices
    colscal = cl.scales['9']['qual']['Set1']

    plotdata = [scatter(
            coords[inds],
            mode='markers',
            marker=dict(
                size=4,
                color=colscal[k % len(colscal)], # set color to an array/list
                #                                  of desired values
                opacity=1),
            name="Cluster {}".format(k),
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

    :param celldata: a Rocks object
    """
    if celldata.Xpca is None or celldata.Xpca.shape[1] < 3:
        print("Need 3 PCs. Calculating now.")
        celldata.pca(3)
    return genericplot(celldata, celldata.Xpca[:,-3:])

def genericwrongplot(celldata, coords, yhat, labels=None):
    """Plot figure with incorrectly classified points highlighted
    
    :param celldata: Rocks object
    :param coords: a (N, 2) or (N, 3) shaped array with coordinates to plot
    :param yhat: (N, 1) shaped array of predicted/guessed y values
    :param labels: (optional) list of axis titles
    """
    import colorlover as cl
    import plotly.graph_objs as go


    y = celldata.y
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
    """Generate a 3d PCA figure with incorrectly classified points highlighted

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
    """Generate a figure for some embedding of Rocks data

    :param celldata: a Rocks object
    :param coords: (N, 2) or (N, 3) shaped coordinates of the embedded data 
    :param genes: list of gene indices
    """
    import colorlover as cl
    import plotly.graph_objs as go

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
    geneexpr = np.round(geneexpr*7/geneexpr.max(axis=0)).astype(int)
    numgenes = geneexpr.shape[1]
    genenames = ["Gene {}".format(genes[i]) if celldata.genes is None \
                 else celldata.genes[i] for i in range(numgenes)]
    
    plotbygene = [scatter(
            coords,
            mode='markers',
            marker=dict(
                size=4,
                color=genescal[geneexpr[:,i]],
                ),
            name=genenames[i],
            hoverinfo="name",
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
        
    updatemenus = [dict(type="buttons", buttons=buttons, direction="left",  x=0.1, y=1.1, xanchor="left", yanchor="top", pad={'r':10, 't': 10},)]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        updatemenus=updatemenus,
        annotations=[
            dict(text='View:', x=0.01, y=1.065, xref='paper', yref='paper', align='left', showarrow=False)],
        showlegend=True,
    )

    return go.Figure(data=plotbyclust+plotbygene, layout=layout)
