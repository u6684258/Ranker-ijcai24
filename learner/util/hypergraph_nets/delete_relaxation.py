from typing import Dict
from typing import List

from util.hypergraph_nets import Node, Hyperedge
from util.hypergraph_nets.hypergraph_view import HypergraphView
from util.mdpsim_api import STRIPSProblem


class DeleteRelaxationHypergraphView(HypergraphView):
    """
    Delete-Relaxation Hypergraph view of a STRIPS problem where:
      - A node corresponds with a single proposition
      - A hyperedge corresponds with a relaxed action, connecting the
        preconditions to the additive effects
    """

    def __init__(self, problem: STRIPSProblem):
        super().__init__(problem)

        # Each node corresponds to a single proposition
        self._nodes = frozenset([str(i).replace('Proposition', '') for i in self.problem.mdpsim_problem.propositions])
        self._node_to_idx: Dict[Node, int] = {
            node: idx for idx, node in enumerate(self.nodes)
        }

        # Each hyperedge corresponds to a relaxed action where the senders
        # are the preconditions and the receivers are the additive effects.
        # Hence, the negative effects are ignored.
        self._hyperedges = [
            Hyperedge(
                name=action.identifier,
                weight=1, # unit cost
                senders=frozenset([str(i).replace('Proposition', '') for i in action.preconditions]),
                receivers=frozenset([str(i).replace('Proposition', '') for i in action.add_effects]),
                # Used to store context of delete-effects for feature mappers
                context={"delete_effects": None},
            )
            for action in self.problem.mdpsim_problem.ground_actions
        ]
        self._hyperedge_to_idx: Dict[Hyperedge, int] = {
            hyperedge: idx for idx, hyperedge in enumerate(self._hyperedges)
        }

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def node_to_idx(self, node: Node) -> int:
        return self._node_to_idx[node]

    @property
    def hyperedges(self) -> List[Hyperedge]:
        return self._hyperedges

    def hyperedge_to_idx(self, hyperedge: Hyperedge) -> int:
        return self._hyperedge_to_idx[hyperedge]
