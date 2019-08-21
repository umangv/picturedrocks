.. py:module:: picturedrocks.markers
.. currentmodule:: picturedrocks.markers

Selecting Markers
-----------------

.. math::
   \DeclareMathOperator{\argmax}{arg max}

PicturedRocks can be used to implement numerous information-theoretic feature selection methods. In both the supervised and unsupervised cases, it is computationally intractable to optimize the true objective functions. We want to find a small set of genes :math:`S`, with :math:`|S| < n` such that, in the supervised case :math:`I(S; y)` is maximized and in the unsupervised case :math:`H(S)` is maximized. (For more information, see our paper)

Because these are computationally intractable, we have implemented the following approximations to :math:`\argmax_{x_i} I(S \cup \{x_i\}; y)` (the ideal supervised objective function)

- The Mutual Information Maximization (MIM) is a univariate feature selection method---it does not consider the interactions between variables and only considers features in isolation.
   .. math::
      \displaystyle J_\text{mim}(x_i) = I(x_i; y)


- The Joint Mutual Information (JMI) algorithm proposed by Yang and Moody uses a diminishing penalty on redundancy. However, it only penalizes relevant redundancy which, we explain in our paper, is desirable.
   .. math::
      \displaystyle J_\text{jmi}(x_i) = I(x_i; y) - \frac{1}{|S|} \sum_{x_j\in S} I(x_i; x_j; y)


- The (CIFE) algorithm proposed by Lin and Tang (and independently by others) does not diminish its redundancy penalty (which it applies only to relevant redundancy, as desired). Although it can penalize redundancy somewhat aggressively, it is the easiest to give theoretical analysis for (see our paper).
   .. math::
      \displaystyle J_\text{cife}(x_i) = I(x_i; y) - \sum_{x_j\in S} I(x_i; x_j; y)



Mutual information
==================

Before running any mutual information based algorithms, we need a discretized
version of the gene expression matrix, with a limited number of discrete
values (because we do not make any assumptions about the distribution of gene
expression). Such data is stored in
:class:`~InformationSet`, but by default, we suggest
using :func:`~makeinfoset` to generate such an object. The :func:`~makeinfoset` function uses the recursive quantile transform :func:`~mutualinformation.infoset.quantile_discretize`. 

Iterative Feature Selection
+++++++++++++++++++++++++++

All information-theoretic feature selection methods in PicturedRocks are
greedy algorithms. In general, they implement the abstract class
:class:`~picturedrocks.markers.mutualinformation.IterativeFeatureSelection` class. See :ref:`sup_mi` and :ref:`unsup_mi` below for specific algorithms.


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
.. autoclass:: InformationSet
   :members:
.. autoclass:: SparseInformationSet
   :members:
.. autofunction:: makeinfoset
.. autofunction:: picturedrocks.markers.mutualinformation.infoset.quantile_discretize
