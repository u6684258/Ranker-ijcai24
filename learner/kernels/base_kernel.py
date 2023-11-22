import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from representation import CGraph


Histogram = Dict[int, int]
NO_EDGE = -2

""" Base class for graph kernels """


class WlAlgorithm(ABC):
    def __init__(self, iterations: int, prune: int) -> None:
        self._train = True

        # hashes neighbour multisets of colours
        self._hash = {}

        # prune if self._train_histogram[col] <= count
        self._prune = prune

        # number of wl iterations
        self.iterations = iterations

        # counters during evaluation of hit and missed colours
        self._hit_colours = 0
        self._missed_colours = 0

        return

    @abstractmethod
    def compute_histograms(self, graphs: List[CGraph]) -> Dict[CGraph, Histogram]:
        raise NotImplementedError

    def get_hash(self) -> Dict[str, int]:
        """Return hash dictionary with compact keys for cpp"""
        ret = {}
        for k in self._hash:
            key = str(k)
            for symbol in [")", "(", " "]:
                key = key.replace(symbol, "")
            ret[key] = self._hash[k]
        return ret

    def _get_hash_value(self, colour) -> int:
        if isinstance(colour, tuple) and len(colour) == 1:
            colour = colour[0]
        if self._train:
            if colour not in self._hash:
                self._hash[colour] = len(self._hash)
            return self._hash[colour]
        else:
            if colour in self._hash:
                self._hit_colours += 1
                return self._hash[colour]
            else:
                self._missed_colours += 1
                return -1

    def _prune_hash(self, histograms):
        inverse_hash = {self._hash[col]: col for col in self._hash}

        # get histogram over all train graphs
        train_histogram = {}
        for G, histogram in histograms.items():
            for col_hash, cnt in histogram.items():
                col = inverse_hash[col_hash]
                if col not in train_histogram:
                    train_histogram[col] = 0
                train_histogram[col] += cnt

        # prune hash
        new_hash = {}
        old_colour_hash_to_new_hash = {}
        for col, old_col_hash in self._hash.items():
            if train_histogram[col] <= self._prune:
                del train_histogram[col]
                continue
            new_col_hash = len(new_hash)
            new_hash[col] = new_col_hash
            old_colour_hash_to_new_hash[old_col_hash] = new_col_hash
        self._hash = new_hash

        # prune from train set
        ret_histograms = {}
        for G, histogram in histograms.items():
            new_histogram = {}
            for old_col_hash, cnt in histogram.items():
                if old_col_hash not in old_colour_hash_to_new_hash:
                    continue  # total count too small
                new_histogram[old_colour_hash_to_new_hash[old_col_hash]] = cnt
            ret_histograms[G] = new_histogram

        return ret_histograms

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False
        self._hit_colours = 0
        self._missed_colours = 0

    def get_hit_colours(self) -> int:
        return self._hit_colours
    
    def get_missed_colours(self) -> int:
        return self._missed_colours

    def get_x(
        self, graphs: CGraph, histograms: Optional[Dict[CGraph, Histogram]] = None
    ) -> np.array:
        """Explicit feature representation
        O(nd) time; n x d output
        """

        n = len(graphs)
        d = len(self._hash)
        X = np.zeros((n, d))

        if histograms is None:
            histograms = self.compute_histograms(graphs)
        else:
            histograms = histograms

        for i, G in enumerate(graphs):
            histogram = histograms[G]
            for j in histogram:
                if 0 <= j and j < d:
                    X[i][j] = histogram[j]

        return X

    def get_k(self, graphs: CGraph, histograms: Dict[CGraph, Histogram]) -> np.array:
        """Implicit feature representation
        O(n^2d) time; n x n output
        """

        n = len(graphs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k = 0

                histogram_i = histograms[graphs[i]]
                histogram_j = histograms[graphs[j]]

                common_colours = set(histogram_i.keys()).intersection(set(histogram_j.keys()))
                for c in common_colours:
                    k += histogram_i[c] * histogram_j[c]

                K[i][j] = k
                K[j][i] = k

        return K

    @property
    def n_colours_(self) -> int:
        return len(self._hash)
