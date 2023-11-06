from stripsHgn.Hgn import Hgn
from .loss import LOSS
from .mpnn import MPNNPredictor
from .elmpnn import ELMPNNPredictor, ELMPNNRankerPredictor, ELMPNNBatchedRankerPredictor, ELMPNNBatchedProbPredictor, \
  ELMPNNBatchedCoordRankerPredictor
from stripsHgn.HgnRanker import PlanRanker

GNNS = {
  "MPNN": MPNNPredictor,
  "RGNN": ELMPNNPredictor,
  "RGNNRANK": ELMPNNRankerPredictor,
  "RGNNBATRANK": ELMPNNBatchedRankerPredictor,
  "RGNNBATCOORDRANK": ELMPNNBatchedCoordRankerPredictor,
  "RGNNBATPROB": ELMPNNBatchedProbPredictor,
  "HGNNRANK": PlanRanker,
  "HGNN": Hgn,
}
