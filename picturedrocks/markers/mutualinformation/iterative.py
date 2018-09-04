
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

from abc import ABC, abstractmethod
import numba as nb
import numpy as np
import scipy.sparse


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


class MIMixin:
    def __init__(self, infoset):
        assert infoset.has_y, "Information Set must have target labels"
        self.infoset = infoset
        self.S = []
        self.base_score = (
            self.infoset.entropy_wrt(np.arange(0))
            + self.infoset.entropy(np.array([-1]))
            - self.infoset.entropy_wrt(np.array([-1]))
        )
        self.penalty = np.zeros(len(self.base_score))
        self.score = self.base_score[:]


class EntropyMixin:
    def __init__(self, infoset):
        self.infoset = infoset
        self.S = []
        self.base_score = self.infoset.entropy_wrt(np.arange(0))
        self.penalty = np.zeros(len(self.base_score))
        self.score = self.base_score.copy()


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


class CIFE(MIMixin, IterativeFeatureSelection):
    def add(self, ind):
        self.S.append(ind)
        penalty_delta = (
            self.base_score
            + self.infoset.entropy(np.array([ind]))
            - self.infoset.entropy_wrt(np.array([ind]))
            - self.infoset.entropy(np.array([-1, ind]))
            + self.infoset.entropy_wrt(np.array([-1, ind]))
        )
        self.penalty += penalty_delta
        self.score = self.base_score - self.penalty
        self.score[self.S] = float("-inf")

    def remove(self, ind):
        self.S.remove(ind)
        penalty_delta = (
            self.base_score
            + self.infoset.entropy(np.array([ind]))
            - self.infoset.entropy_wrt(np.array([ind]))
            - self.infoset.entropy(np.array([-1, ind]))
            + self.infoset.entropy_wrt(np.array([-1, ind]))
        )
        self.penalty -= penalty_delta
        self.score = self.base_score - self.penalty
        self.score[self.S] = float("-inf")


class JMI(MIMixin, IterativeFeatureSelection):
    def add(self, ind):
        self.S.append(ind)
        penalty_delta = (
            self.base_score
            + self.infoset.entropy(np.array([ind]))
            - self.infoset.entropy_wrt(np.array([ind]))
            - self.infoset.entropy(np.array([-1, ind]))
            + self.infoset.entropy_wrt(np.array([-1, ind]))
        )
        self.penalty += penalty_delta
        self.score = self.base_score - (self.penalty / len(self.S))
        self.score[self.S] = float("-inf")

    def remove(self, ind):
        self.S.remove(ind)
        penalty_delta = (
            self.base_score
            + self.infoset.entropy(np.array([ind]))
            - self.infoset.entropy_wrt(np.array([ind]))
            - self.infoset.entropy(np.array([-1, ind]))
            + self.infoset.entropy_wrt(np.array([-1, ind]))
        )
        self.penalty -= penalty_delta
        self.score = self.base_score - (self.penalty / len(self.S))
        self.score[self.S] = float("-inf")


class MIM(MIMixin, UnivariateMixin, IterativeFeatureSelection):
    pass


class UniEntropy(EntropyMixin, UnivariateMixin, IterativeFeatureSelection):
    pass


class CIFEUnsup(EntropyMixin, IterativeFeatureSelection):
    def add(self, ind):
        self.S.append(ind)
        penalty_delta = (
            self.base_score  # H(x_i)
            + self.infoset.entropy(np.array([ind]))  # H(x_j)
            - self.infoset.entropy_wrt(np.array([ind]))  # H(x_i, x_j)
        )
        self.penalty += penalty_delta
        self.score = self.base_score - self.penalty
        self.score[self.S] = float("-inf")

    def remove(self, ind):
        self.S.remove(ind)
        penalty_delta = (
            self.base_score  # H(x_i)
            + self.infoset.entropy(np.array([ind]))  # H(x_j)
            - self.infoset.entropy_wrt(np.array([ind]))  # H(x_i, x_j)
        )
        self.penalty -= penalty_delta
        self.score = self.base_score - self.penalty
        self.score[self.S] = float("-inf")

    pass
