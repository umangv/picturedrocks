# Copyright © 2018 Anna Gilbert, Alexander Vargo, Umang Varma
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
from abc import ABC, abstractmethod


class IterativeFeatureSelection(ABC):
    @abstractmethod
    def __init__(self, infoset):
        self.infoset = infoset
        self.score = None
        self.S = []

    @abstractmethod
    def add(self, ind):
        pass

    @abstractmethod
    def remove(self, ind):
        pass

    def autoselect(self, n_feats):
        for i in range(n_feats):
            best = np.argmax(self.score)
            self.add(best)


class UnivariateMixin:
    def add(self, ind):
        self.S.append(ind)
        self.score[ind] = float("-inf")

    def remove(self, ind):
        self.S.remove(ind)
        self.score[ind] = self.base_score[ind]

    def autoselect(self, n_feats):
        nbest = np.argpartition(self.score, -n_feats)[-n_feats:]
        nbest = nbest[np.argsort(self.score[nbest])[::-1]]
        self.S.extend(nbest)
        self.score[nbest] = float("-inf")
        return nbest
