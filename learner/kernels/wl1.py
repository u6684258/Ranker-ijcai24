from typing import Optional, Dict
from tqdm import tqdm
from .base_kernel import *


class ColourRefinement(WlAlgorithm):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_hash(self) -> Dict[str, int]:
        """Return hash dictionary with compact keys for cpp"""
        ret = {}
        for k in self._hash:  # Dict[str, int]
            key = str(k)
            for symbol in [")", "(", " "]:
                key = key.replace(symbol, "")
            ret[key] = self._hash[k]
        return ret

    def compute_histograms(self, graphs: List[CGraph]) -> Dict[CGraph, Histogram]:
        """Read graphs and return histogram.

        self._train value determines if new colours are stored or not
        """

        histograms = {}

        # compute colours and hashmap from training data
        for G in tqdm(graphs):
            cur_colours = {}
            histogram = {}

            def store_colour(colour):
                nonlocal histogram
                colour_hash = self._get_hash_value(colour)
                if colour_hash not in histogram:
                    histogram[colour_hash] = 0
                histogram[colour_hash] += 1

            # collect initial colours
            for u in G.nodes:
                # initial colour is feature of the node
                colour = G.nodes[u]["colour"]
                cur_colours[u] = self._get_hash_value(colour)
                # assert colour in self._hash and colour>=0, colour
                store_colour(colour)

            # WL iterations
            for itr in range(self.iterations):
                new_colours = {}
                for u in G.nodes:
                    # edge label WL variant
                    neighbour_colours = []
                    for v in G[u]:
                        colour_node = cur_colours[v]
                        colour_edge = G.edges[(u, v)]["edge_label"]
                        neighbour_colours.append((colour_node, colour_edge))
                    neighbour_colours = sorted(neighbour_colours)
                    colour = tuple([cur_colours[u]] + neighbour_colours)
                    new_colours[u] = self._get_hash_value(colour)
                    store_colour(colour)

                cur_colours = new_colours

            # store histogram of graph colours
            histograms[G] = histogram

        if self._train and self._prune > 0:
            histograms = self._prune_hash(histograms)

        return histograms
