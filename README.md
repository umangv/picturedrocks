# PicturedRocks Single Cell Analysis Tool

PicturedRocks is a python package that implements some single cell analysis algorithms that we are studying. Currently, we implement two marker selection algorithms:

1. 1-bit Compressed Sensing algorithms based on [Conrad, et al. BMC bioinformatics '17]
2. variants of mutual information based algorithms (e.g., the "minimum Redundance Maximum Relevance" algorithm [Peng, et al. IEEE TPAMI '05])

## Usage

To install the latest GitHub version of PicturedRocks, do an "editable" installation of PicturedRocks:
```
git clone git@github.com:umangv/picturedrocks.git
cd picturedrocks
pip install -e .
```

PicturedRocks in compatible with `scanpy` and uses its `AnnData` objects. Most methods require cluster labels to be loaded. 

```python
from picturedrocks.read import read_clusts, process_clusts
adata = read_clusts(adata, "clust_labels.csv")
adata = process_clusts(adata)
```

More detailed information can be found on the [online documentation](https://picturedrocks.rtfd.io/).

## Code Style

Pull requests are welcome. Please use [numpy-style docstrings](https://sphinxcontrib-napoleon.rtfd.io/) and format your code with [black](https://black.rtfd.io).

## Copyright

Copyright © 2017, 2018 Anna Gilbert, Alexander Vargo, Umang Varma

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
