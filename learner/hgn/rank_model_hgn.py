import json
from typing import List

import numpy as np
from torch import nn, Tensor
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from hgn.hypergraph_nets.hypergraphs import HypergraphsTuple
from hgn.hypergraph_nets.models import EncodeProcessDecode


def _get_debug_dict(hparams):
    return {
        "latent_size": hparams["latent_size"],
        "batch_size": hparams["batch_size"],
        "num_steps": hparams["num_steps"],
        "receiver_k": hparams["receiver_k"],
        "sender_k": hparams["sender_k"],
        "hidden_size": hparams["hidden_size"],
    }
class HgnRankModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hparams = {}
        self.hparams = params
        self._prediction_mode = False
        self.show_countdown = 0


        # Setup HGN
        self.base_network = EncodeProcessDecode(
            receiver_k=self.hparams["receiver_k"],
            sender_k=self.hparams["sender_k"],
            hidden_size=self.hparams["hidden_size"],
            global_input_size=self.hparams["global_feature_mapper_cls"].input_size(),
            edge_input_size=self.hparams["hyperedge_feature_mapper_cls"].input_size(),
            node_input_size=self.hparams["node_feature_mapper_cls"].input_size(),
            global_output_size=self.hparams["latent_size"],
            last_relu=False
        )

        self.ranker = DirectRanker(
            latent_size=self.hparams["latent_size"],
            antisymmetric_activation=torch.sigmoid,
        )

        # Log hyperparameters
        print(
            "hparams:\n"
            f"{json.dumps(_get_debug_dict(self.hparams), indent=2)}"
        )

    def setup_prediction_mode(self):
        self._prediction_mode = True
        self.ranker.setup_prediction_mode()

    def get_num_parameters(self) -> int:
        """ Count number of weight parameters """
        # https://stackoverflow.com/a/62764464/13531424
        # e.g. to deal with case of sharing layers
        params = sum(dict((p.data_ptr(), p.numel()) for p in self.parameters() if p.requires_grad).values())
        return params

    def forward(self, hypergraphs: HypergraphsTuple):
        """
        Run the forward pass through the network

        Parameters
        ----------
        hypergraph_left, hypergraph_right: HypergraphsTuple, one or more hypergraphs

        Returns
        -------
        Output rank order prediction
        """
        # print(hypergraph_left)
        # self.show_countdown += 1
        left = self.base_network(
            hypergraph=hypergraphs,
            steps=self.hparams["num_steps"],
            pred_mode=self._prediction_mode,
        )
        if not self._prediction_mode:
            # if self.show_countdown % 1000 == 0:
            #     print(f"Encodes: {left[-1].globals} ")
            results = []
            polaritys = []
            for i in range(len(left)):
                result, polarity = self.ranker(hypergraphs, left[i].globals)
                results.append(result)
                polaritys.append(polarity)
            return results, polaritys

        else:
            pred, polarity = self.ranker(hypergraphs, left[-1].globals)
            # if self.show_countdown % 100 == 0:
            #     _log.debug(f"hgn output: {left[-1].globals.detach().numpy()}, ranker output: {pred.detach().numpy()}")

            return pred, polarity

    def shift_heu(self, h, scale=1e3, shift=1e5):
        result = h + shift
        # print(f"result: {result}")
        assert (2147483647 > result).all() and (
                    result > 0).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all() and (result > 0).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result

class DirectRanker(nn.Module):
    def __init__(self, latent_size,
                 antisymmetric_activation=torch.tanh):
        super().__init__()
        self._ranker = nn.Linear(latent_size, 1, bias=False)
        self._activation = antisymmetric_activation
        self._prediction_mode = False
    def setup_prediction_mode(self):
        self._prediction_mode = True

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
                polarity = torch.ones(diff.size(0))
                if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                    print(f"Warning: Encodings are very close to each other: {diff}")

                if torch.sum(torch.abs(result) - 0.5) / diff.shape[0] < 1e-3:
                    print(f"Warning: Classification is close to 0: {result}")
            return result, polarity

    def trace(self, features):
        return torch.jit.trace(self, features[0])


    def shift_heur(self, h, scale=1e5, shift=1e3):
        result = h + shift
        # print(f"result: {result}")
        assert (2147483647 > result).all() and (
                result > 0).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all() and (result > 0).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result