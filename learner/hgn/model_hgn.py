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
class HgnModel(nn.Module):
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
            global_output_size=1,
            last_relu=True
        )

        # Log hyperparameters
        print(
            "hparams:\n"
            f"{json.dumps(_get_debug_dict(self.hparams), indent=2)}"
        )

    def setup_prediction_mode(self):
        self._prediction_mode = True

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
                result, polarity = (left[i].globals, hypergraphs.y_value)
                results.append(result)
                polaritys.append(polarity)
            return results, polaritys

        else:
            pred, polarity = (left[-1].globals.detach().numpy(), hypergraphs.y_value)

        return pred, polarity


class HGNLoss():
  def __init__(self) -> None:
    self._criterion = torch.nn.MSELoss()
  def calc_avg_loss(self,
                    preds: List[Tensor],
                    target: List[Tensor],
                    ):
    """
    Calculates average loss for a criterion over multiple predictions
    """
    sum_index = 1
    start_index = 0
    accum_loss = self._criterion(preds[start_index], target[start_index])
    for pass_idx in range(start_index + 1, len(preds)):
      loss = self._criterion(
        preds[pass_idx], target[pass_idx]
      )
      accum_loss += loss
      sum_index += 1

    return accum_loss / float(sum_index)

  def forward(self, preds: List[Tensor],
                    target: List[Tensor],
                    ):

    return self.calc_avg_loss(preds, target)
