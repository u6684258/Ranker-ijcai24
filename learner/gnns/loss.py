from typing import List

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import MSELoss, BCEWithLogitsLoss


""" Weights loss higher for inadmissible estimates """
class WeightedMSELoss:
  def __init__(self, weight: float=2) -> None:
    self.weight = weight

  def forward(self, input: Tensor, target: Tensor):
    lt = (input < target).nonzero().squeeze(1)
    gt = (input > target).nonzero().squeeze(1)
    loss = 0
    if len(lt) > 0:
      lt_loss = F.mse_loss(
        input=torch.index_select(input,0,lt),
        target=torch.index_select(target,0,lt)
        )
      loss += lt_loss
    if len(gt) > 0:
      gt_loss = F.mse_loss(
        input=torch.index_select(input,0,gt),
        target=torch.index_select(target,0,gt)
      )
      loss += self.weight*gt_loss
    return loss


""" Loss function from Mehdi Samadi, Ariel Felner, and Jonathan Schaeﬀer, AAAI 2008 """
class PenaltyEnhancedMSELoss():
  def __init__(self, a: float=1, b:float=1) -> None:
    self.a = a
    self.b = b

  def forward(self, input: Tensor, target: Tensor):
    e = (input - target)**2
    e = (self.a + 1/(1 + torch.exp(-self.b * e))) * e
    loss = torch.sum(e) / len(e)
    return loss

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


LOSS = {
  "mse": MSELoss,
  "wmse": WeightedMSELoss,
  "pemse": PenaltyEnhancedMSELoss,
  "hgnloss": HGNLoss,
}

