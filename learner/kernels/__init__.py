from .wrapper import KernelModelWrapper
from .wl1 import ColourRefinement
from .wl2 import WL2
from .gwl2 import GWL2
from .lwl2 import LWL2

GRAPH_FEATURE_GENERATORS = {
    "1wl": ColourRefinement,
    "2wl": WL2,
    "2gwl": GWL2,
    "2lwl": LWL2,
}
