from .mpnn import MPNNPredictor
from .elmpnn import ELMPNNPredictor, ELMPNNRankerPredictor

GNNS = {
  "MPNN": MPNNPredictor,
  "RGNN": ELMPNNPredictor,
  "RGNNRANK": ELMPNNRankerPredictor,
}
