import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
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

""" This file contains three variants of classes:
    1. GNN layer module classes which are stacked within a GNN class
    1. GNN classes themselves
    2. a class which acts as an interface containing a GNN and Representation object
"""


def construct_mlp(in_features: int, out_features: int, n_hid: int) -> torch.nn.Module:
    return Sequential(
        Linear(in_features, n_hid),
        ReLU(),
        Linear(n_hid, out_features),
    )


class RGNNLayer(Module):
    def __init__(self, in_features: int, out_features: int, n_edge_labels: int, aggr: str):
        super(RGNNLayer, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(n_edge_labels):
            self.convs.append(LinearConv(in_features, out_features, aggr=aggr).jittable())
        self.root = Linear(in_features, out_features, bias=True)
        return

    def forward(self, x: Tensor, list_of_edge_index: List[Tensor]) -> Tensor:
        x_out = self.root(x)
        for i, conv in enumerate(self.convs):  # bottleneck; difficult to parallelise efficiently
            x_out += conv(x, list_of_edge_index[i])
        return x_out


class LinearConv(MessagePassing):
    propagate_type = {"x": Tensor}

    def __init__(self, in_features: int, out_features: int, aggr: str) -> None:
        super().__init__(aggr=aggr)
        self.f = Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # propagate_type = {'x': Tensor }
        x = self.f(x)
        x = self.propagate(edge_index=edge_index, x=x, size=None)
        return x


class GLN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_feat = params["in_feat"]
        self.out_feat = params["out_feat"]
        self.nhid = params["nhid"]
        self.aggr = params["aggr"]
        self.n_edge_labels = params["n_edge_labels"]
        self.nlayers = params["nlayers"]
        self.rep_type = params["rep"]

        if params["pool"] == "max":
            self.pool = global_max_pool
        elif params["pool"] == "mean":
            self.pool = global_mean_pool
        elif params["pool"] == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError
        self.emb = torch.nn.Linear(self.in_feat, self.nhid)
        # self.convs = torch.Sequen
        print(f"in feat: {self.in_feat}, nhid: {self.nhid}")
        self.conv1 = RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)
        self.conv2 = RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)
        self.conv3 = RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)
        self.conv4 = RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)
        self.leaky_relu = LeakyReLU(negative_slope=0.01)
        self.relu = ReLU()
        # self.conv4 = RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)
        # self.mlp_h = construct_mlp(in_features=self.nhid, n_hid=self.nhid, out_features=self.out_feat)
        self.mlp_h = nn.Sequential(
            Linear(self.nhid, self.nhid),
            ReLU(),
            Linear(self.nhid, 1),
        )

    def forward(self, x: Tensor, edge_index, batch: Optional[Tensor]):
        x = self.emb(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.pool(x, batch)
        h = self.mlp_h(x)
        return h.squeeze(1)

class RGNN(nn.Module):
    """
    The class can be compiled with jit or the new pytorch-2. However, pytorch-geometric
    has yet to provide compiling for GNNs with variable sized graph inputs.
    """

    def __init__(self, params) -> None:
        super().__init__()
        self.in_feat = params["in_feat"]
        self.out_feat = params["out_feat"]
        self.nhid = params["nhid"]
        self.aggr = params["aggr"]
        self.n_edge_labels = params["n_edge_labels"]
        self.nlayers = params["nlayers"]
        self.rep_type = params["rep"]
        self.rep = None
        self.device = None
        self.batch = False

        if params["pool"] == "max":
            self.pool = global_max_pool
        elif params["pool"] == "mean":
            self.pool = global_mean_pool
        elif params["pool"] == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError

        self.initialise_layers()

        return

    @abstractmethod
    def create_layer(self) -> None:
        raise NotImplementedError

    def initialise_layers(self) -> None:
        self.emb = torch.nn.Linear(self.in_feat, self.nhid)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(self.create_layer())
        self.mlp_h = construct_mlp(in_features=self.nhid, n_hid=self.nhid, out_features=self.out_feat)
        return

    def create_layer(self):
        return RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)

    def node_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, list_of_edge_index)
            x = F.leaky_relu(x)

        return x

    def graph_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        x = self.node_embedding(x, list_of_edge_index, batch)
        x = self.pool(x, batch)
        return x

    def forward(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        x = self.graph_embedding(x, list_of_edge_index, batch)
        h = self.mlp_h(x)
        h = h.squeeze(1)
        return h

    def name(self) -> str:
        return type(self).__name__


class Model(nn.Module):
    """
    A wrapper for a GNN which contains the GNN, additional informations beyond hyperparameters,
    and helpful methods such as I/O and providing an interface for planners to call as a heuristic
    evaluator.
    """

    def __init__(self, params=None, jit=False) -> None:
        super().__init__()
        if params is not None:
            self.model = None
            self.jit = jit
            self.rep_type = params["rep"]
            self.rep = None
            self.device = None
            self.batch = False
            self.create_model(params)
        if self.jit:
            self.model = torch.jit.script(self.model)
        return
    
    def set_eval(self) -> None:
        self.model.eval()
        return

    def lifted_state_input(self) -> bool:
        return self.rep.lifted

    def dump_model_stats(self) -> None:
        print(f"Model name: RGNN")
        print(f"Device:", self.device)
        print(f"Number of parameters:", self.get_num_parameters())
        print(f"Number of layers:", self.model.nlayers)
        print(f"Number of hidden units:", self.model.nhid)
        return

    def load_state_dict_into_gnn(self, model_state_dict) -> None:
        """Load saved weights"""
        self.model.load_state_dict(model_state_dict)

    def forward(self, data):
        return self.model.forward(data.x, data.edge_index, data.batch)

    def embeddings(self, data):
        return self.model.graph_embedding(data.x, data.edge_index, data.batch)

    def forward_from_embeddings(self, embeddings):
        x = self.model.mlp(embeddings)
        # x = x.squeeze(1)
        return x

    def initialise_readout(self):
        if self.jit:
            self.model.mlp = torch.jit.script(
                construct_mlp(
                    in_features=self.model.nhid,
                    n_hid=self.model.nhid,
                    out_features=self.model.out_feat,
                )
            )
        else:
            self.model.mlp = construct_mlp(
                in_features=self.model.nhid,
                n_hid=self.model.nhid,
                out_features=self.model.out_feat,
            )
        return

    def update_representation(self, domain_pddl: str, problem_pddl: str, args, device):
        self.rep: Representation = REPRESENTATIONS[self.rep_type](domain_pddl, problem_pddl)
        self.rep.convert_to_pyg()
        self.device = device
        return

    def update_device(self, device):
        self.device = device
        return

    def batch_search(self, batch: bool):
        self.batch = batch
        return

    def print_weights(self) -> None:
        weights = self.state_dict()
        for weight_group in weights:
            print(weight_group)
            print(weights[weight_group])
        return

    def get_num_parameters(self) -> int:
        """Count number of weight parameters"""
        # https://stackoverflow.com/a/62764464/13531424
        # e.g. to deal with case of sharing layers
        params = sum(
            dict((p.data_ptr(), p.numel()) for p in self.parameters() if p.requires_grad).values()
        )
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return params

    def get_num_zero_parameters(self) -> int:
        """Count number of parameters that are zero after training"""
        zero_weights = 0
        for p in self.parameters():
            if p.requires_grad:
                zero_weights += torch.sum(torch.isclose(p.data, torch.zeros_like(p.data)))
        return zero_weights

    def print_num_parameters(self) -> None:
        print(f"number of parameters: {self.get_num_parameters()}")
        return

    def set_zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def create_model(self, params):
        if params["model"] == "gnn-new":
            self.model = GLN(params)
        else:
            self.model = RGNN(params)



    def h(self, state: State) -> float:
        with torch.no_grad():
            x, edge_index = self.rep.state_to_tgraph(state)
            x = x.to(self.device)
            for i in range(len(edge_index)):
                edge_index[i] = edge_index[i].to(self.device)
            h = self.model.forward(x, edge_index, None)
            h = round(h.item())
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
            hs_all = np.rint(hs_all)
            hs_all = hs_all.astype(int).tolist()
            return hs_all

    def __call__(self, node_or_list_nodes):  # call on Pyperplan search
        if self.batch:
            states = [n.state for n in node_or_list_nodes]
            h = self.h_batch(states)  # list of states
        else:
            state = node_or_list_nodes.state
            h = self.h(state)  # single state
        return h

    def name(self) -> str:
        return self.model.name()
