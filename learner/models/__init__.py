from .mpnn import MPNNPredictor
from .elmpnn import ELMPNNPredictor, ELMPNNRankerPredictor, ELMPNNBatchedRankerPredictor

GNNS = {
  "MPNN": MPNNPredictor,
  "RGNN": ELMPNNPredictor,
  "RGNNRANK": ELMPNNRankerPredictor,
  "RGNNBATRANK": ELMPNNBatchedRankerPredictor,
}
