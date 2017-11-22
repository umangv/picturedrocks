# PicturedRocks Single Cell Analysis Tool

PicturedRocks is a python package that implements some single cell analysis algorithms that we are studying. Currently, we implement two marker selection algorithms:

1. a 1-bit Compressed Sensing algorithm based on [Conrad, et al. BMC bioinformatics '17]
2. a mutual information based "Maximum Relevance minimum Redundance" algorithm based on [Peng, et al. IEEE TPAMI '05]

## Usage

To install, put the `picturedrocks` directory in your python path (e.g., in the current working directory).

To create a `Rocks` object, you need `X` (the gene expression matrix, a numpy array of shape (N, P) where the dataset has N cells and P rows), `y` (the cluster label vector, an numpy array of shape (N, 1) and dtype `int`), and, optionally, a list of gene names `genes`.

```python
from picturedrocks import Rocks
rocks = Rocks(X, y, genes=genes)
```

More detailed information can be found on the [online documentation](https://picturedrocks.github.io/docs/).


## Copyright

Copyright © 2017 Anna Gilbert, Alexander Vargo, Umang Varma

PicturedRocks is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PicturedRocks is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PicturedRocks.  If not, see <http://www.gnu.org/licenses/>.
