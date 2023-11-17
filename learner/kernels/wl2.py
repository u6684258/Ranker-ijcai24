from typing import Optional, Dict
from itertools import product
from tqdm import tqdm
from .base_kernel import *

""" 2-FWL algorithm, not 2-WL """


class Wl2(WlAlgorithm):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_hash(self) -> Dict[str, int]:
        """Return hash dictionary with compact keys for cpp"""
        ret = {}
        for k in self._hash:
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

            n_nodes = len(G.nodes)
            assert set(G.nodes) == set(range(n_nodes))

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

    def eval(self):
        super().eval()
        self._train_histogram = None

    @property
    def n_colours_(self) -> int:
        return len(self._hash)
