import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings

from models.gnn import Model
from planning import Proposition, State
from representation import REPRESENTATIONS, Representation
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from abc import ABC, abstractmethod
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, Dropout, LeakyReLU, BatchNorm1d
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import Tensor
from typing import Optional, List, FrozenSet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.nn.conv import (
#     RGCNConv,
#     FastRGCNConv,
# )  # (slow and/or mem inefficient)

class LossModel(Model):
    """
    A wrapper for a GNN which contains the GNN, additional informations beyond hyperparameters,
    and helpful methods such as I/O and providing an interface for planners to call as a heuristic
    evaluator.
    """

    def __init__(self, params=None, jit=False) -> None:
        super().__init__(params, jit)

    def h(self, state: State) -> float:
        with torch.no_grad():
            x, edge_index = self.rep.state_to_tgraph(state)
            x = x.to(self.device)
            for i in range(len(edge_index)):
                edge_index[i] = edge_index[i].to(self.device)
            h = self.model.forward(x, edge_index, None)
            h = self.shift_heu(h)[0]
            return h

    def h_batch(self, states: List[State]) -> List[float]:
        with torch.no_grad():
            data_list = []
            for state in states:
                x, edge_index = self.rep.state_to_tgraph(state)
                data_list.append(Data(x=x, edge_index=edge_index))
            loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
            hs_all = []
            for data in loader:
                data = data.to(self.device)
                hs = self.model.forward(data.x, data.edge_index, data.batch)
                hs = hs.detach().cpu().numpy()  # annoying error with jit
                hs_all.append(hs)
            hs_all = np.concatenate(hs_all)
            # hs_all = np.rint(hs_all)
            hs_all = self.shift_heu(hs_all).tolist()
            return hs_all


    def shift_heu(self, h, scale=1e3, shift=1):
        result = h + shift
        # print(f"result: {result}")
        # # and (result > 0).all(),
        # assert (2147483647 > result).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result
