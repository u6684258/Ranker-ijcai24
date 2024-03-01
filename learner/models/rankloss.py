from typing import List

import torch
from torch import Tensor
from torch_geometric.data import Data


class RankLoss():

    def forward(self, preds: Tensor,
                    target: Data,
                    ):
        loss = torch.tensor([0.0]).to(preds.device)
        coord_x = target.coord_x
        coord_y = target.coord_y
        remove_indices = []
        for i in range(torch.max(target.coord_x)):
            optimal_idx = ((target.coord_x == i) & (target.coord_y == 0)).nonzero(as_tuple=True)[0]
            remove_indices.append(optimal_idx)
            if i == torch.max(target.coord_x):
                next_idx = target.coord_x.shape[0]
            else:
                next_idx = ((target.coord_x == i+1) & (target.coord_y == 0)).nonzero(as_tuple=True)[0]
            to_compare = []
            for j, p in enumerate(remove_indices):
                if j == len(remove_indices) - 1:
                    if p+1!=next_idx:
                        to_compare.append(preds[p+1:next_idx])
                else:
                    if p+1!=remove_indices[j+1]:
                        to_compare.append(preds[p+1:remove_indices[j+1]])

            if not to_compare:
                continue
            loss += torch.sum(torch.log(1+torch.exp(preds[optimal_idx] - torch.concatenate(to_compare))))

        if loss == 0:
            loss = torch.sum(preds - preds)
        return loss