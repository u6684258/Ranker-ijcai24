from .loss import LOSS
from .mpnn import MPNNPredictor
from .elmpnn import ELMPNNPredictor, ELMPNNRankerPredictor, ELMPNNBatchedRankerPredictor, ELMPNNBatchedProbPredictor

GNNS = {
  "MPNN": MPNNPredictor,
  "RGNN": ELMPNNPredictor,
  "RGNNRANK": ELMPNNRankerPredictor,
  "RGNNBATRANK": ELMPNNBatchedRankerPredictor,
  "RGNNBATPROB": ELMPNNBatchedProbPredictor,
}
