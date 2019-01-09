Selecting Markers
-----------------

PicturedRocks current implements two categories of marker selection algorithms:
 * mutual information-based algorithms 
 * 1-bit compressed sensing based algorithms

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

Iterative Feature Selection
+++++++++++++++++++++++++++

All information-theoretic feature selection methods in PicturedRocks are
greedy algorithms. In general, they implement the abstract class
:any:`IterativeFeatureSelection` class. See :ref:`sup_mi` and :ref:`unsup_mi`
for specific algorithms.

.. automodule:: picturedrocks.markers.mutualinformation.iterative

.. autoclass:: IterativeFeatureSelection
    :members:

.. _sup_mi:

Supervised Feature Selection
****************************
.. autoclass:: MIM
.. autoclass:: CIFE
.. autoclass:: JMI

.. _unsup_mi:

Unsupervised Feature Selection
******************************
.. autoclass:: UniEntropy
.. autoclass:: CIFEUnsup

Auxiliary Classes and Methods
+++++++++++++++++++++++++++++
.. autoclass:: picturedrocks.markers.InformationSet
.. autoclass:: picturedrocks.markers.SparseInformationSet
.. autofunction:: picturedrocks.markers.makeinfoset

Interactive Marker Selection
============================

.. autoclass:: picturedrocks.markers.interactive.InteractiveMarkerSelection
.. autofunction:: picturedrocks.markers.interactive.cife_obj
