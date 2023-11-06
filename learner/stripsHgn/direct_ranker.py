from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from representation.hypergraph_nets.hypergraphs import HypergraphsTuple

class DirectRanker(nn.Module):
    def __init__(self, latent_size,
                 antisymmetric_activation=torch.tanh):
        super().__init__()
        self._ranker = nn.Linear(latent_size, 1, bias=False)
        self._activation = antisymmetric_activation
        self._prediction_mode = False
    def setup_prediction_mode(self):
        self._prediction_mode = True

    def shift_heur(self, h, scale=1e2, shift=1e4):
        result = h + shift
        # print(f"result: {result}")
        assert (2147483647 > result).all() and (
                result > 0).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all() and (result > 0).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result

    def forward(self, data, encodes):
        # print(states)
        if self._prediction_mode:
            result = self.shift_heur(self._ranker(encodes).squeeze(1).detach().numpy())
            return result, 0
        else:
            encodes_xy = torch.concatenate([encodes, data.x_coord.reshape([-1, 1]), data.y_coord.reshape([-1, 1])], dim=1)
            unique = torch.unique(data.x_coord)
            split_by_x = [encodes_xy[(data.x_coord.reshape([-1,]) == i)] for i in unique]
            diff = []
            for s in split_by_x:
                sort_s = s[s[:, -1].sort()[1]]
                diff.append(sort_s[1:, :-2] - sort_s[0, :-2])
            diff = torch.concatenate(diff, dim=0)
            assert diff.size(0) + unique.size(0) == encodes.size(0)
            result = self._activation(self._ranker(diff)).squeeze(1)
            with torch.no_grad():
                polarity = torch.ones(diff.size(0)).float()
                if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                    print(f"Warning: Encodings are very close to each other: {diff}")

                if torch.sum(torch.abs(result)) / diff.shape[0] < 1e-3:
                    print(f"Warning: Classification is close to 0: {result}")
            return result, polarity

    def trace(self, features):
        return torch.jit.trace(self, features[0])


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, nb_layers=2, size_hidden=50,
                 hidden_activation=functional.relu,
                 final_activation=lambda x: x):
        super().__init__()
        assert nb_layers > 0, nb_layers
        assert size_hidden > 0, size_hidden

        self.layers = []
        for no_layer in range(nb_layers):
            layer = nn.Linear(
                input_dim if no_layer == 0 else size_hidden,
                output_dim if no_layer == nb_layers - 1 else size_hidden)
            self.add_module(f"fc_{no_layer}", layer)
            self.layers.append(layer)

            self.layers.append(
                hidden_activation if no_layer != nb_layers - 1 else
                final_activation)

    def forward(self, *states):
        x = torch.cat(states, 1)
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def factory(dataset, **kwargs):
        input_dim = (sum(dataset.input_dim)
                     if isinstance(dataset.input_dim, Iterable) else
                     dataset.input_dim)
        output_dim = kwargs.pop("output_dim", dataset.output_dim)

        return FFN(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )


class FFNGraph(nn.Module):
    """
    A FFN module for Plan Ranker. Input a hypergraph and output an embedding vector.
    """
    def __init__(self, input_dims, output_dim, nb_layers=2, size_hidden=50,
                 hidden_activation=functional.relu,
                 final_activation=lambda x: x):
        """
        :param input_dims: list of four numbers:
                input_dims[0]: number of propositions
                input_dims[1]: shape of proposition vectors
                input_dims[2]: number of actions
                input_dims[3]: shape of action vectors
        :param output_dim: latent size (input dim to the ranker)
        :param nb_layers: number of affine layers
        :param size_hidden: hidden layers' input/output size
        :param hidden_activation: activation function of hidden layers
        :param final_activation: last activation function
        """
        super().__init__()
        assert nb_layers > 0, nb_layers
        assert size_hidden > 0, size_hidden

        self.layers = []
        self.input_dim = input_dims[0] * input_dims[1] + input_dims[2] * input_dims[3]
        # self.input_dim = 110
        for no_layer in range(nb_layers):
            layer = nn.Linear(
                self.input_dim if no_layer == 0 else size_hidden,
                output_dim if no_layer == nb_layers - 1 else size_hidden)
            self.add_module(f"fc_{no_layer}", layer)
            self.layers.append(layer)

            self.layers.append(
                hidden_activation if no_layer != nb_layers - 1 else
                final_activation)

    def forward(
            self, hypergraph: HypergraphsTuple, steps: int, pred_mode: bool = False
    ):
        x = torch.cat([hypergraph.nodes.reshape(-1), hypergraph.edges.reshape(-1)])
        print(x.size())
        print(self.input_dim)
        for layer in self.layers:
            x = layer(x)

        new_graph = hypergraph.replace(globals=x)
        return [new_graph for _ in range(10)]
