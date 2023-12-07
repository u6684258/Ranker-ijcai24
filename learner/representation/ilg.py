from .base_class import *
from planning.translate.pddl import Atom, NegatedAtom, Truth
from enum import Enum


class ILG_FEATURES(Enum):
    G = 0  # is positive goal (grounded)
    N = 1  # is negative goal (grounded)
    O = 2  # is object
    S = 3  # is activated (grounded) [only used in GNN]


ENC_FEAT_SIZE = len(ILG_FEATURES)

# additional hard coded colours
ACTIVATED_COLOUR = 0
ACTIVATED_POS_GOAL_COLOUR = 1
ACTIVATED_NEG_GOAL_COLOUR = 2


""" ILG from GenPlan23 submission """


class InstanceLearningGraph(Representation, ABC):
    name = "ilg"
    n_node_features = ENC_FEAT_SIZE
    n_edge_labels = float("inf")  # unbounded because of var size; adapted from input (see below)
    directed = False
    lifted = True

    def __init__(self, domain_pddl: str, problem_pddl: str):
        super().__init__(domain_pddl, problem_pddl)

    def _get_to_coloured_graphs_init_colours(self):
        return {
            ACTIVATED_COLOUR: ACTIVATED_COLOUR,
            ACTIVATED_POS_GOAL_COLOUR: ACTIVATED_POS_GOAL_COLOUR,
            ACTIVATED_NEG_GOAL_COLOUR: ACTIVATED_NEG_GOAL_COLOUR,
        }

    def _compute_graph_representation(self) -> None:
        """TODO: reference definition of this graph representation

        everything is sorted to try to make deterministic
        """

        G = self._create_graph()

        # fd has an =(x, y) predicate
        predicates = sorted([p for p in self.problem.predicates if p.name != "="])
        self.n_predicates = len(predicates)

        # TODO(DZC) option to change n_node_features depending on whether we want to refine
        # predicate colour/encoding or not
        self.n_node_features = ENC_FEAT_SIZE + self.n_predicates

        # objects
        for i, obj in enumerate(sorted(self.problem.objects)):
            G.add_node(obj.name, x=self._one_hot_node(ILG_FEATURES.O.value))  # add object node

        # predicates
        largest_predicate = 0
        for i, pred in enumerate(predicates):
            if pred.name[0] == "=":
                continue
            largest_predicate = max(largest_predicate, len(pred.arguments))
            G.add_node(pred.name, x=self._one_hot_node(ENC_FEAT_SIZE + i))  # add predicate node
        self.largest_predicate = largest_predicate
        self.n_edge_labels = largest_predicate + 1  # add one for -1 edge labels

        # debugging
        tmp = {
            0: "G",
            1: "N",
            2: "O",
            3: "S (not used)",
        }
        for i, pred in enumerate(predicates):
            tmp[ENC_FEAT_SIZE + i] = pred.name
        self.debugging = {}
        for k, v in tmp.items():
            k = str(tuple(self._one_hot_node(k).tolist()))
            self.debugging[k] = v
        self.debugging[0] = "true non-goal"
        self.debugging[1] = "true pos-goal"
        self.debugging[2] = "true neg-goal"

        # goal (state gets dealt with in state_to_tensor)
        if len(self.problem.goal.parts) == 0:
            goals = [self.problem.goal]
        else:
            goals = self.problem.goal.parts
        for fact in sorted(goals):
            assert type(fact) in {Atom, NegatedAtom}

            # may have negative goals
            is_negated = type(fact) == NegatedAtom

            pred = fact.predicate
            args = fact.args
            goal_node = (pred, args)

            if is_negated:
                x = self._one_hot_node(ILG_FEATURES.N.value)
                self._neg_goal_nodes.add(goal_node)
            else:
                x = self._one_hot_node(ILG_FEATURES.G.value)
                self._pos_goal_nodes.add(goal_node)
            G.add_node(goal_node, x=x)  # add fact node

            # connect fact to predicate
            assert pred in G.nodes()
            G.add_edge(u_of_edge=goal_node, v_of_edge=pred, edge_label=-1)
            G.add_edge(v_of_edge=goal_node, u_of_edge=pred, edge_label=-1)

            for k, arg in enumerate(args):
                # connect fact to object
                G.add_edge(u_of_edge=goal_node, v_of_edge=arg, edge_label=k)
                G.add_edge(v_of_edge=goal_node, u_of_edge=arg, edge_label=k)
        # end goal

        assert largest_predicate > 0

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

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._node_to_i:
                x[self._node_to_i[node]][ILG_FEATURES.S.value] = 1
                continue

            # activated proposition does not overlap with a goal
            true_node_i = i
            x[i][ILG_FEATURES.S.value] = 1
            i += 1

            # connect fact to predicate
            append_edge_index[-1].append((true_node_i, self._node_to_i[pred]))
            append_edge_index[-1].append((self._node_to_i[pred], true_node_i))

            # connect fact to objects (different from ILG: node position nodes)
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

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._pos_goal_nodes:
                idx = self._name_to_node[node]
                c_graph.nodes[idx]["colour"] = ACTIVATED_POS_GOAL_COLOUR
                # print(node, idx)
                continue
            elif node in self._neg_goal_nodes:
                idx = self._name_to_node[node]
                c_graph.nodes[idx]["colour"] = ACTIVATED_NEG_GOAL_COLOUR
                continue

            node = new_idx
            new_idx += 1

            # else add node and corresponding edges to graph
            c_graph.add_node(node, colour=ACTIVATED_COLOUR)

            # connect fact to predicate
            assert self._name_to_node[pred] in c_graph.nodes
            c_graph.add_edge(u_of_edge=node, v_of_edge=self._name_to_node[pred], edge_label=-1)
            c_graph.add_edge(v_of_edge=node, u_of_edge=self._name_to_node[pred], edge_label=-1)

            for k, obj in enumerate(args):
                # connect fact to object
                assert self._name_to_node[obj] in c_graph.nodes
                c_graph.add_edge(u_of_edge=node, v_of_edge=self._name_to_node[obj], edge_label=k)
                c_graph.add_edge(v_of_edge=node, u_of_edge=self._name_to_node[obj], edge_label=k)

        return c_graph
