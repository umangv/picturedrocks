.. PicturedRocks documentation master file, created by
   sphinx-quickstart on Sun Nov 19 22:21:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PicturedRocks--Single Cell RNA-seq Analysis Tool
================================================

.. image:: https://travis-ci.com/umangv/picturedrocks.svg?branch=master
    :target: https://travis-ci.com/umangv/picturedrocks
    :alt: Build Status
.. image:: https://readthedocs.org/projects/picturedrocks/badge/?version=latest
    :target: https://picturedrocks.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

PicturedRocks is a python package that implements information-theoretic feature selection algorithms for scRNA-seq analysis. 


Installing
==========

Please ensure you have Python 3.6 or newer and have `numba` and `scikit-learn` installed. The best way to get Python and various dependencies is with Anaconda or Miniconda. Once you have a conda environment, run ``conda install numba scikit-learn``. Then use pip to install PicturedRocks and all additional dependencies::

    pip install picturedrocks

To install the latest code from github, clone our github repository. Once inside the project directory, instal by running ``pip install -e .``. The ``-e`` option will point a symbolic link to the current directory instead of installing a copy on your system. This allows you to update picturedrocks by pulling the latest commit from github.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   FeatureSelectionTutorial.ipynb
   read
   preprocessing
   plot
   markers
   interactive
   performance



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
