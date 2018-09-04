Selecting Markers
-----------------

PicturedRocks current implements two categories of marker selection algorithms:
 * mutual information-based algorithms 
 * 1-bit compressed sensing based algorithms

.. toctree::
    :hidden:

Mutual information
==================

TODO: Explanation of how these work goes here.

Before running any mutual information based algorithms, we need a discretized
version of the gene expression matrix, with a limited number of discrete
values (because we do not make any assumptions about the distribution of gene
expression). Such data is stored in
:class:`picturedrocks.markers.InformationSet`, but by default, we suggest
using :func:`picturedrocks.markers.makeinfoset` to generate such an object
after appropriate normalization

.. autoclass:: picturedrocks.markers.mutualinformation.iterative.MIM
.. autoclass:: picturedrocks.markers.mutualinformation.iterative.CIFE
.. autoclass:: picturedrocks.markers.mutualinformation.iterative.JMI
.. autoclass:: picturedrocks.markers.mutualinformation.iterative.UniEntropy
.. autoclass:: picturedrocks.markers.mutualinformation.iterative.CIFEUnsup

Auxiliary Classes and Methods
+++++++++++++++++++++++++++++
.. autoclass:: picturedrocks.markers.InformationSet
.. autofunction:: picturedrocks.markers.makeinfoset

Interactive Marker Selection
============================

.. autoclass:: picturedrocks.markers.interactive.InteractiveMarkerSelection
.. autofunction:: picturedrocks.markers.interactive.cife_obj
