from .base_class import CGraph, TGraph, Representation
from .slg import StripsLearningGraph
from .dlg import DeleteLearningGraph
from .flg import FdrLearningGraph
from .llg import LiftedLearningGraph
from .ilg import InstanceLearningGraph
from .ilg2 import InstanceLearningGraph2
from .glg import GroundedLearningGraph


REPRESENTATIONS = {
    "slg": StripsLearningGraph,
    "dlg": DeleteLearningGraph,
    "flg": FdrLearningGraph,
    "llg": LiftedLearningGraph,
    "ilg": InstanceLearningGraph,
    "ilg2": InstanceLearningGraph2,
    "glg": GroundedLearningGraph,
}
