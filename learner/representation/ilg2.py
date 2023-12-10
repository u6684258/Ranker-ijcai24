from .base_class import *
from planning.translate.pddl import Atom, NegatedAtom, Truth
from enum import Enum

F_POS_GOAL = 0
T_POS_GOAL = 1
T_NON_GOAL = 2


""" reduce unseen colours """


class InstanceLearningGraph2(Representation, ABC):
    name = "ilg2"
    n_node_features = float("inf")  # to read from domain
    n_edge_labels = float("inf")  # unbounded because of var size; adapted from input (see below)
    directed = False
    lifted = True

    def __init__(self, domain_pddl: str, problem_pddl: str):
        super().__init__(domain_pddl, problem_pddl)

    def _get_to_coloured_graphs_init_colours(self):
        return {k:k for k in range(self.n_node_features)}

    def _compute_graph_representation(self) -> None:
        """TODO: reference definition of this graph representation

        everything is sorted to try to make deterministic
        """

        G = self._create_graph()

        # fd has an =(x, y) predicate
        self.n_predicates = len(self.predicates)

        # TODO(DZC) option to change n_node_features depending on whether we want to refine
        # predicate colour/encoding or not
        self.n_node_features = 1 + self.n_predicates * 3

        # objects
        for i, obj in enumerate(sorted(self.problem.objects)):
            G.add_node(obj.name, colour=0)  # add object node

        # predicates
        largest_predicate = 0
        for i, pred in enumerate(self.predicates):
            largest_predicate = max(largest_predicate, len(pred.arguments))
        self.largest_predicate = largest_predicate
        self.n_edge_labels = largest_predicate  # no longer -1 edge labels
        assert largest_predicate > 0

        # debugging
        self.colour_explanation = {
            0: "ob",  # object
        }
        # TODO make this mapping a bijective function
        for i, pred in enumerate(self.predicates):
            self.colour_explanation[1 + 3 * i + F_POS_GOAL] = f"ug {pred.name}"
            self.colour_explanation[1 + 3 * i + T_POS_GOAL] = f"ag {pred.name}"
            self.colour_explanation[1 + 3 * i + T_NON_GOAL] = f"ap {pred.name}"

        # goal (state gets dealt with in state_to_tensor)
        if len(self.problem.goal.parts) == 0:
            goals = [self.problem.goal]
        else:
            goals = self.problem.goal.parts
        for fact in sorted(goals):
            assert type(fact) in {Atom, NegatedAtom}

            # may have negative goals
            is_negated = type(fact) == NegatedAtom
            if is_negated:
                raise NotImplementedError

            pred = fact.predicate
            args = fact.args
            goal_node = (pred, args)

            colour = 1 + 3 * self.pred_to_idx[pred] + F_POS_GOAL
            G.add_node(goal_node, colour=colour)  # add fact node

            self._pos_goal_nodes.add(goal_node)

            for k, arg in enumerate(args):
                # connect fact to object
                G.add_edge(u_of_edge=goal_node, v_of_edge=arg, edge_label=k)
                G.add_edge(v_of_edge=goal_node, u_of_edge=arg, edge_label=k)
        # end goal

        # map node name to index
        self._node_to_i = {}
        for i, node in enumerate(G.nodes):
            self._node_to_i[node] = i
        self.G = G

        return

    def str_to_state(self, s) -> List[Tuple[str, List[str]]]:
        """Used in dataset construction to convert string representation of facts into a (pred, [args]) representation"""
        state = []
        for fact in s:
            fact = fact.replace(")", "").replace("(", "")
            toks = fact.split()
            if toks[0] == "=":
                continue
            if len(toks) > 1:
                state.append((toks[0], toks[1:]))
            else:
                state.append((toks[0], ()))
        return state

    def state_to_tensor(self, state: List[Tuple[str, List[str]]]) -> TGraph:
        """States are represented as a list of (pred, [args])"""
        x = self.x.clone()
        edge_indices = self.edge_indices.copy()
        i = len(x)

        to_add = sum(len(fact[1]) + 1 for fact in state)
        x = torch.nn.functional.pad(x, (0, 0, 0, to_add), "constant", 0)
        append_edge_index = {i: [] for i in range(-1, self.largest_predicate)}

        for fact in state:
            pred = fact[0]
            args = fact[1]

            if len(pred) == 0:
                continue

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal
            if node in self._node_to_i:
                x[self._node_to_i[node]][1 + 3 * self.pred_to_idx[pred] + T_POS_GOAL] = 1
                continue

            # activated proposition does not overlap with a goal
            true_node_i = i
            x[i][1 + 3 * self.pred_to_idx[pred] + T_NON_GOAL] = 1
            i += 1

            # connect fact to objects 
            for k, arg in enumerate(args):
                append_edge_index[k].append((true_node_i, self._node_to_i[arg]))
                append_edge_index[k].append((self._node_to_i[arg], true_node_i))

        for i, append_edges in append_edge_index.items():
            edge_indices[i] = torch.hstack((edge_indices[i], torch.tensor(append_edges).T)).long()

        return x, edge_indices

    def state_to_cgraph(self, state: List[Tuple[str, List[str]]]) -> CGraph:
        """States are represented as a list of (pred, [args])"""
        c_graph = self.c_graph.copy()
        new_idx = len(self._name_to_node)

        for fact in state:
            pred = fact[0]
            args = fact[1]

            if len(pred) == 0:
                continue

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal Atom
            if node in self._pos_goal_nodes:
                idx = self._name_to_node[node]
                assert c_graph.nodes[idx]["colour"] == 1 + 3 * self.pred_to_idx[pred] + F_POS_GOAL
                c_graph.nodes[idx]["colour"] = 1 + 3 * self.pred_to_idx[pred] + T_POS_GOAL
                # print(c_graph.nodes[idx]["colour"])
                # print(node, idx)
                continue

            node = new_idx
            new_idx += 1

            # else add node and corresponding edges to graph
            c_graph.add_node(node, colour=1 + 3 * self.pred_to_idx[pred] + T_NON_GOAL)

            for k, obj in enumerate(args):
                # connect fact to object
                assert self._name_to_node[obj] in c_graph.nodes
                c_graph.add_edge(u_of_edge=node, v_of_edge=self._name_to_node[obj], edge_label=k)
                c_graph.add_edge(v_of_edge=node, u_of_edge=self._name_to_node[obj], edge_label=k)

        return c_graph
