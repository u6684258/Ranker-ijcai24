from .base_class import CGraph, TGraph, Representation
from .slg import StripsLearningGraph
from .dlg import DeleteLearningGraph
from .flg import FdrLearningGraph
from .llg import LiftedLearningGraph
from .glg import GroundedLearningGraph
from .hypergraph_nets.hypergraphs import HypergraphsTuple

REPRESENTATIONS = {
  "slg": StripsLearningGraph,
  "dlg": DeleteLearningGraph,
  "flg": FdrLearningGraph,
  "llg": LiftedLearningGraph,
  "glg": GroundedLearningGraph,
  "hgn": HypergraphsTuple,
  "hgn_ranker": HypergraphsTuple,
}

