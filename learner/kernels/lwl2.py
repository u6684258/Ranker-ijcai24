from typing import Optional, Dict
from itertools import product
from tqdm import tqdm
from .base_kernel import *

""" 2-LWL """


class LWL2(WlAlgorithm):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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

            n_nodes = len(G.nodes)
            assert set(G.nodes) == set(range(n_nodes))

            raise NotImplementedError

            tuples = list(product(G.nodes, G.nodes))

            # collect initial colours
            for tup in tuples:
                u, v = tup

                # initial colour is feature of the node
                c_u = G.nodes[u]["colour"]
                c_v = G.nodes[v]["colour"]
                # graph is undirected so this equals (v, u) in G.edges
                edge = (u, v)
                is_edge = (edge in G.edges)
                if is_edge:
                    edge_colour = G.edges[edge]["edge_label"]
                else:
                    edge_colour = '_'  # no edge
                # the more general k-wl algorithm colours by looking at colour-isomorphism
                colour = (c_u, c_v, edge_colour)

                cur_colours[tup] = self._get_hash_value(colour)
                assert colour in self._hash, colour
                store_colour(colour)

            # WL iterations
            for itr in range(self.iterations):
                new_colours = {}
                for tup in tuples:
                    u, v = tup

                    # k-wl does not care about graph structure after initial colours
                    neighbour_colours = []
                    for w in G.nodes:
                        neighbour_colours.append((cur_colours[(u, w)], cur_colours[(w, v)]))

                    # equation-wise, neighbour colours is a multiset of tuple colours
                    neighbour_colours = sorted(neighbour_colours)
                    colour = tuple([cur_colours[tup]] + neighbour_colours)
                    new_colours[tup] = self._get_hash_value(colour)
                    store_colour(colour)

                cur_colours = new_colours

            # store histogram of graph colours
            histograms[G] = histogram

        if self._train:
            histograms = self._prune_hash(histograms)

        return histograms
