import os
from typing import List

import numpy as np
from torch import nn
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from models.gnn import Model


class RankModel(Model):
    def __init__(self, params):
        super().__init__(params)
        self.model.mlp_h = nn.Sequential(
            nn.Linear(self.model.nhid, self.model.out_feat),
        )
        self.model.ranker = nn.Linear(params["out_feat"], 1, bias=False)
        self.model.ranker_act = nn.Sigmoid()

    def forward(self, data):
        with torch.no_grad():
            assert torch.sum(data.p_idx) - data.p_idx[0] * data.p_idx.shape[0] == 0
            # print(data.problem)
        encodes = self.model.forward(data.x, data.edge_index, data.batch)

        encodes_xy = torch.concatenate([encodes, data.coord_x.reshape([-1, 1]), data.coord_y.reshape([-1, 1])], dim=1)
        unique = torch.unique(data.coord_x)
        split_by_x = [encodes_xy[data.coord_x == i] for i in unique]
        diff = []
        for s in split_by_x:
            # init state, no states worse than it
            if s[0, -2] == 0:
                continue
            sort_s = s[s[:, -1].sort()[1]]
            # print(torch.logical_and(data.coord_x == (s[0, -2].item()-1), data.coord_y == 0))
            diff.append(sort_s[1:, :-2] - sort_s[0, :-2])
            last_layer_best = encodes_xy[torch.logical_and(data.coord_x == (s[0, -2].item()-1), data.coord_y == 0)]
            diff.append((last_layer_best[0, :-2] - sort_s[0, :-2]).reshape([1, -1]))
        diff = torch.concatenate(diff, dim=0)
        assert diff.size(0) == encodes.size(0) - 1

        result = self.model.ranker_act(self.model.ranker(diff)).squeeze(1)
        with torch.no_grad():
            polarity = torch.ones(diff.size(0))
            if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                print(f"Warning: Encodings are very close to each other: {diff}")

            if torch.sum(torch.abs(result) - 0.5) / diff.shape[0] < 1e-3:
                print(f"Warning: Classification is close to 0: {result}")
        return result, polarity

    def h(self, state) -> float:
        with torch.no_grad():
            x, edge_index = self.rep.state_to_tgraph(state)
            x = x.to(self.device)
            for i in range(len(edge_index)):
                edge_index[i] = edge_index[i].to(self.device)
            h = self.model.forward(x, edge_index, None)
            h = self.model.ranker(h).detach().cpu().numpy()
            h = self.shift_heu(h)[0]
            return h

    def h_batch(self, states: List) -> List[float]:
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
                hs = self.model.ranker(hs).detach().cpu().numpy()  # annoying error with jit
                hs_all.append(hs)
            hs_all = np.concatenate(hs_all)
            hs_all = self.shift_heu(hs_all).tolist()
            return hs_all

    def shift_heu(self, h, scale=1, shift=1):
        result = h + shift
        # print(f"result: {result}")
        # # and (result > 0).all(),
        # assert (2147483647 > result).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.rint(result * scale).astype(int)
        assert (2147483647 > result).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result
