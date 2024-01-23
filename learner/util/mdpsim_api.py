import os
from numbers import Number
import random
from types import ModuleType
from typing import TypeVar, List, Dict, Tuple, Any

from util.prob_dom_meta import get_domain_meta, get_problem_meta
from util.state_reprs import CanonicalState, strip_parens

"""
Wrapper of MDPSIM grounded problem instances.
based on ASNETS
"""

# Proposition - i.e. fact, atom
Proposition = TypeVar("Proposition", bound=str)
State = CanonicalState

EXP_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
TMP_DIR = os.path.join(EXP_ROOT, "tmp")


class PDDLLoadError(Exception):
    """PDDL parse exception"""


def parse_problem_args(mdpsim, domain_pddl: str,
                       problem_pddls: List[str]) -> Any:
    """Parse a problem from a given MDPSim module.

    Args:

    Raises:
        PDDLLoadError: If the problem could not be loaded.

    Returns:
        Any: The problem.
    """
    pddl_name_map = {}
    success = mdpsim.parse_file(domain_pddl)
    cur_problems = set(mdpsim.get_problems().keys())
    if not success:
        raise PDDLLoadError('Could not parse %s' % domain_pddl)
    for problem in problem_pddls:
        success = mdpsim.parse_file(problem)
        if not success:
            raise PDDLLoadError('Could not parse %s' % domain_pddl)
        name = set(mdpsim.get_problems().keys()) - cur_problems
        assert len(name) == 1
        pddl_name_map[problem] = list(name)[0]
        cur_problems = set(mdpsim.get_problems().keys())

    return pddl_name_map


class STRIPSProblem:
    def __init__(self, domain_pddl: str, problem_pddls: List[str]):
        self.domain_pddl = domain_pddl
        self.problem_pddls = problem_pddls

        import mdpsim

        self.mdpsim: ModuleType = mdpsim
        self.pddl_name_map = parse_problem_args(self.mdpsim,
                                                 domain_pddl,
                                                 problem_pddls)

        self.mdpsim_problems = mdpsim.get_problems()


    def change_problem(self, problem, solution_file=None):
        self.mdpsim_problem = self.mdpsim_problems[self.pddl_name_map[problem]]
        self.domain = self.mdpsim_problem.domain
        self.domain_meta = get_domain_meta(self.mdpsim_problem.domain)
        self.problem_meta = get_problem_meta(self.mdpsim_problem,
                                             self.domain_meta)
        self.max_receivers, self.max_senders = self.get_hgn_data(self.domain_pddl)
        self.solution_file = solution_file

        self.act_ident_to_mdpsim_act: Dict[str, Any] = {
            strip_parens(a.identifier): a
            for a in self.mdpsim_problem.ground_actions
        }

        if solution_file:
            # Heuristic value buffer
            self._state_to_heuristic: Dict[State, Number] = {}
            self.plan_to_state_heuristics(self.initial_state, self.solution_file)

    @property
    def domain_name(self) -> str:
        return self.domain.name

    @property
    def name(self) -> str:
        return self.problem_meta.name

    @property
    def initial_state(self) -> CanonicalState:
        mdpsim_init = self.mdpsim_problem.init_state()
        cstate_init = CanonicalState.from_mdpsim(mdpsim_init,
                                                 self,
                                                 prev_cstate=None,
                                                 prev_act=None,
                                                 is_init_cstate=True)

        return cstate_init
    @property
    def state_to_heuristic(self) -> Dict[State, Number]:
        return self._state_to_heuristic

    def is_goal_state(self, state: CanonicalState) -> bool:
        """ Whether the given state is a goal state """
        return state.is_goal


    def record_state_heuristics(self, states_heu_pairs: List[Tuple[State, Number]]):
        for state, heu in states_heu_pairs:
            self._state_to_heuristic[state] = heu

    def get_successors(self, state: CanonicalState) -> List[State]:
        mdpsim_state = state.to_mdpsim(self)
        successors = []
        for bound_act, applicable in state.acts_enabled:
            if not applicable:
                continue
            else:
                act_ident = bound_act.unique_ident
                mdpsim_action = self.act_ident_to_mdpsim_act[act_ident]
                new_mdpsim_state = self.mdpsim_problem.apply(
                    mdpsim_state, mdpsim_action)
                new_cstate = CanonicalState.from_mdpsim(new_mdpsim_state,
                                                        self,
                                                        prev_cstate=state,
                                                        prev_act=bound_act,
                                                        is_init_cstate=False)
                successors.append(new_cstate)

        return successors


    def plan_to_state_heuristics(self, init_state, plan_file):
        # Read the plan file
        with open(plan_file, "rb") as plan_file:
            actions_str = plan_file.readlines()[:-1]
            plan = [action.decode("UTF-8").strip() for action in actions_str]

        trajectory = {str(init_state): init_state}
        hStar = {str(init_state): len(plan)}

        accum_diff = 0
        current_state = init_state.to_mdpsim(self)
        prev_state = None
        for idx, action_name in enumerate(plan):
            # Apply action in the current state
            action_name = action_name.replace("(", "").replace(")", "")
            action = self.act_ident_to_mdpsim_act[action_name]
            prev_state = current_state
            current_state = self.mdpsim_problem.apply(
                    current_state, action)

            cstate = CanonicalState.from_mdpsim(current_state,
                                self,
                                prev_cstate=prev_state,
                                prev_act=action,
                                is_init_cstate=False)

            # Create new state-value pair
            remaining_plan_length = len(plan) - (idx + 1)

            if str(cstate) in trajectory.keys():
                diff = hStar[str(cstate)] - remaining_plan_length
                accum_diff += diff
                hStar[str(cstate)] -= diff

                for k, v in hStar.items():
                    if k == str(cstate):
                        continue
                    hStar[k] -= diff
                    if hStar[k] < hStar[str(cstate)]:
                        hStar[k] = -1


            else:
                trajectory[str(cstate)] = cstate

                hStar[str(cstate)] = remaining_plan_length

        # Check current state is a goal state and the number of pairs
        assert self.is_goal_state(CanonicalState.from_mdpsim(current_state,
                                                        self,
                                                        prev_cstate=prev_state,
                                                        prev_act=action,
                                                        is_init_cstate=False))

        traj = []
        for i in trajectory.keys():
            if hStar[i] >=0:
                traj.append((trajectory[i], hStar[i]))

        assert len(traj) == len(plan) + 1 - accum_diff
        self.record_state_heuristics(traj)

        return len(plan)


    def generate_extended_state_dataset(self, origin_state_pairs, step=1, toy_set=False):

        successor_set = {}
        by_index_state = {}
        new_pair_set = {}
        max_heu = max(origin_state_pairs.values())

        for state, heu in origin_state_pairs.items():
            # state = self.to_partial_state(state)
            by_index_state[heu] = state
            successor_set[state] = [pair for pair in self.get_successors(state)]

        for heu, state in by_index_state.items():
            # init state, no worse state
            if heu == max_heu:
                continue
            # append parent state
            new_pair_set[state] = [by_index_state[heu + 1]]
            # append parent's successors except this state
            for i in range(heu + 1, min(max_heu + 1, heu + 1 + step)):
                new_pair_set[state] += successor_set[by_index_state[i]]
            new_pair_set[state].remove(state)
            if toy_set and len(new_pair_set[state]) > 4:
                new_pair_set[state] = random.sample(new_pair_set[state], 4)

        return new_pair_set

    def get_hgn_data(self, domain_pddl):
        max_senders = 0
        max_receivers = 0
        with open(domain_pddl) as f:
            lines = f.read().split(":")

            for line in lines:
                if "precondition" in line:
                    num_senders = line.count(")")
                    # remove (and ...)
                    if num_senders > 1:
                        num_senders -= 1
                    # print(num_senders)
                    # print(line)
                    if num_senders > max_senders:
                        max_senders = num_senders

                elif "effect" in line:
                    # remove delete effects
                    num_receivers = 0
                    left = 0
                    for c in line:
                        if c == "(":
                            left += 1
                        elif c == ")" and left > 0:
                            num_receivers += 1
                            left -= 1

                    num_receivers -= 1
                    num_receivers -= 2 * line.count("(not ")

                    # print(num_receivers)
                    # print(line)
                    if num_receivers > max_receivers:
                        max_receivers = num_receivers

        return max_receivers, max_senders



if __name__ == "__main__":
    domain = "blocksworld"
    problem = "p03.pddl"
    # problem = "p-l2-c10-s1.pddl"
    domain_f = f"/home/ming/PycharmProjects/goose/benchmarks/ipc23/{domain}/domain.pddl"
    problem_f = f"/home/ming/PycharmProjects/goose/benchmarks/ipc23/{domain}/training/easy/{problem}"
    solution_f = f"/home/ming/PycharmProjects/goose/benchmarks/ipc23/solutions/{domain}/training/easy/{problem.replace('.pddl', '.plan')}"
    problem = STRIPSProblem(domain_f, problem_f, solution_f)
    # print(problem.get_successors(problem.initial_state))
    # action = Operator(name='(unstack b3 b2)',
    #                   preconditions=frozenset({'(on b3 b2)', '(clear b3)', '(handempty)'}),
    #                   add_effects=frozenset({'(holding b3)', '(clear b2)'}),
    #                   del_effects=frozenset({'(handempty)', '(clear b3)', '(on b3 b2)'}))
    # new_state = action.apply(problem.initial_state)
    # print(problem.get_state_heuristic(new_state))
    # print(new_state)
    # print(problem.to_partial_state(new_state))
    # print(problem.get_successors(problem.to_partial_state(problem.initial_state)))
    # dataset = problem.generate_one_step_pair_dataset(problem.state_to_heuristic)
    # print(problem.goals)
    print(problem.initial_state)
    print(problem.state_to_heuristic)
    # print(problem.get_successors(problem.initial_state))
    # for state, l in dataset.items():
    #     print(state)
    #     print(l)
    # goose_s = "[('at-ferry', ['l0']), ('empty-ferry', []), ('at', ['c0', 'l1']), ('at', ['c1', 'l0']), ('at', ['c2', 'l1']), ('at', ['c3', 'l1']), ('at', ['c4', 'l1']), ('at', ['c5', 'l1']), ('at', ['c6', 'l0']), ('at', ['c7', 'l0']), ('at', ['c8', 'l1']), ('at', ['c9', 'l1'])]"
    # goose_s = "[('', []), ('clear', ['b2']), ('', []), ('', []), ('on-table', ['b1']), ('on', ['b2', 'b1']), ('holding', ['b3'])]"
    # print(problem.goose_state_to_pyperplan(goose_s))
    # print(problem.generate_extended_state_dataset(problem.state_to_heuristic))
