from .wrapper import KernelModelWrapper
from .wl1 import ColourRefinement
from .wl2 import Wl2

GRAPH_FEATURE_GENERATORS = {
    "wl": ColourRefinement,
    "2wl": Wl2,
}
