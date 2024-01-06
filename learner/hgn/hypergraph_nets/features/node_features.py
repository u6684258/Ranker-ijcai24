from typing import FrozenSet, List, Type

from hgn.hypergraph_nets import Node, Hyperedge, Number
from hgn.hypergraph_nets.features import NodeFeatureMapper
from util.mdpsim_api import Proposition


class PropositionInStateAndGoal(NodeFeatureMapper):
    """
    Each proposition is mapped to a 2-dimensional feature vector where:
      [
        whether proposition is in current state,
        whether proposition is in goal state
      ]
    """

    def __init__(
        self,
        current_state: FrozenSet[Proposition],
        goal_state: FrozenSet[Proposition],
    ):
        self._current_state = current_state
        self._goal_state = goal_state

    def node_to_feature(self, node: Node) -> List[Number]:
        x = [
            1 if node in self._current_state else 0,
            1 if node in self._goal_state else 0,
        ]
        return x

    @classmethod
    def name(cls) -> str:
        return "simple"

    @classmethod
    def input_size(cls) -> int:
        return 2


# Used for command line args
NODE_FEATURE_MAPPERS = {
    feature_mapper_cls.name(): feature_mapper_cls
    for feature_mapper_cls in (PropositionInStateAndGoal,)
}

DEFAULT_NODE_FEATURE_MAPPER = PropositionInStateAndGoal.name()
assert DEFAULT_NODE_FEATURE_MAPPER in NODE_FEATURE_MAPPERS


def get_node_feature_mapper(mapper: str) -> Type[NodeFeatureMapper]:
    if mapper not in NODE_FEATURE_MAPPERS:
        raise ValueError(f"Unsupported node feature mapper {mapper}")

    return NODE_FEATURE_MAPPERS[mapper]
