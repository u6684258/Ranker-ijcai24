import os
import shutil
import subprocess
import time
from numbers import Number
from pathlib import Path
from typing import NamedTuple, FrozenSet, TypeVar, List, Dict, Tuple

from pyperplan.pyperplan.planner import get_domain_and_task
from pyperplan.pyperplan.pddl.pddl import Domain as PyperplanDomain
from pyperplan.pyperplan.task import Task as PyperplanTask, Operator
from util.search import fd_general_cmd

"""
Wrapper of Pyperplan grounded problem instances.
based on STRIPS-HGN (https://github.com/williamshen-nz/STRIPS-HGN)
"""

# Proposition - i.e. fact, atom
Proposition = TypeVar("Proposition", bound=str)
State = FrozenSet[Proposition]

EXP_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
TMP_DIR = os.path.join(EXP_ROOT, "tmp")


class STRIPSProblem:
    def __init__(self, domain_pddl: str, problem_pddl: str, solution_file: str = None):
        self.domain_pddl = domain_pddl
        self.problem_pddl = problem_pddl
        self.solution_file = solution_file

        domain, task = get_domain_and_task(domain_pddl, problem_pddl)

        self._pyperplan_domain: PyperplanDomain = domain
        self._pyperplan_task: PyperplanTask = task

        # Mapping of proposition to a unique ID
        self._proposition_to_idx = {}
        self._propositions = []

        for idx, proposition in enumerate(self._pyperplan_task.facts):
            self._proposition_to_idx[proposition] = idx
            self._propositions.append(proposition)

        # Pyperplan only supports unit cost actions
        self._actions = self._pyperplan_task.operators

        self.name_to_action: Dict[str, Operator] = {
            action.name: action for action in self.actions
        }

        self._ground_props = self.initial_state - self._pyperplan_task.facts

        # Heuristic value buffer
        self._state_to_heuristic: Dict[State, Number] = {}
        if self.solution_file:
            self.plan_to_state_heuristics(self.initial_state, self.solution_file)
        else:
            self.get_state_heuristic(self.initial_state)
    @property
    def domain_name(self) -> str:
        return self._pyperplan_domain.name

    @property
    def name(self) -> str:
        return self._pyperplan_task.name

    @property
    def initial_state(self) -> State:
        return self._pyperplan_task.initial_state

    @property
    def goals(self) -> State:
        return self._pyperplan_task.goals

    @property
    def propositions(self) -> List[Proposition]:
        return self._propositions

    @property
    def actions(self) -> List[Operator]:
        return self._actions

    @property
    def number_of_propositions(self) -> int:
        return len(self.propositions)

    @property
    def state_to_heuristic(self) -> Dict[State, Number]:
        return self._state_to_heuristic

    def is_goal_state(self, state: FrozenSet[Proposition]) -> bool:
        """ Whether the given state is a goal state """
        return self.goals.issubset(state)

    def to_partial_state(self, state):
        return state & self._pyperplan_task.facts

    def to_full_state(self, state):
        return state | self._ground_props

    def record_state_heuristics(self, states_heu_pairs: List[Tuple[State, Number]]):
        for state, heu in states_heu_pairs:
            self._state_to_heuristic[state] = heu

    def get_successors(self, state: State) -> List[Tuple[Operator, State]]:
        return self._pyperplan_task.get_successor_states(state)

    def plan_to_state_heuristics(self, init_state, plan_file):
        # Read the plan file
        with open(plan_file, "rb") as plan_file:
            actions_str = plan_file.readlines()[:-1]
            plan = [action.decode("UTF-8").strip() for action in actions_str]

        current_state = init_state
        trajectory: List[Tuple[State, Number]] = [
            (current_state, len(plan))
        ]
        for idx, action_name in enumerate(plan):
            # Apply action in the current state
            action = self.name_to_action[action_name]
            current_state = action.apply(current_state)

            # Create new state-value pair
            remaining_plan_length = len(plan) - (idx + 1)
            trajectory.append((current_state, remaining_plan_length))

        # Check current state is a goal state and the number of pairs
        assert self.is_goal_state(current_state)
        assert len(trajectory) == len(plan) + 1
        self.record_state_heuristics(trajectory)

        return len(plan)


    def make_problem_file(self, state: State):
        state = self.to_full_state(state)
        file_dir = os.path.join(TMP_DIR, "tmp_problem.pddl")
        with open(self.problem_pddl, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "(:init" in line:
                    start_index = i + 1
                elif "(:goal" in line:
                    end_index = i - 1
            new_init = sorted([prop + "\n" for prop in state])
            lines = lines[:start_index] + new_init + lines[end_index:]
            with open(file_dir, "w") as f1:
                f1.writelines(lines)
        return file_dir
    def get_state_heuristic(self, state: State):
        # If the state's heuristic already exists, return the heuristic value
        if state in self._state_to_heuristic.keys():
            return self._state_to_heuristic[state]

        # Else, try to compute the state's heuristic by solving the problem
        # starting from the state and ends in the same goal
        else:
            print("Calculating Heuristic using fd ...")
            Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
            result_file = os.path.join(TMP_DIR, "tmp_result")
            log_file = os.path.join(TMP_DIR, "tmp_log")
            problem_file = self.make_problem_file(state)
            search_cmd = fd_general_cmd(self.domain_pddl, problem_file, result_file)
            st = time.time()
            with open(log_file, 'w') as out_fp:
                rv = subprocess.Popen(search_cmd,
                                      cwd=EXP_ROOT,
                                      stdout=out_fp,
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True,
                                      )
                rv.wait(timeout=70)

            et = time.time()

            print(f"true time: {et - st} seconds")
            result = 999999
            success = False
            with open(log_file, 'r') as stdout_fp:
                out_text = stdout_fp.read()
                if "Solution found." not in out_text:
                    print(search_cmd)
                    print(out_text)
                    print(f"Failed to compute the heuristic of state {state} within "
                          f"time limit, treating the state as a dead end")

                    self.record_state_heuristics([(state, result)])
                else:
                    success = True
            if success:
                result = self.plan_to_state_heuristics(state, result_file)
            try:
                shutil.rmtree(TMP_DIR)
            except OSError:
                pass

            return result

    def generate_extended_state_dataset(self, origin_state_pairs, step=1):
        successor_set = {}
        by_index_state = {}
        new_pair_set = {}
        max_heu = max(origin_state_pairs.values())

        for state, heu in origin_state_pairs.items():
            # state = self.to_partial_state(state)
            by_index_state[heu] = state
            successor_set[state] = [pair[1] for pair in self.get_successors(state)]

        for heu, state in by_index_state.items():
            # init state, no worse state
            if heu == max_heu:
                continue
            # append parent state
            new_pair_set[state] = [by_index_state[heu + 1]]
            # final state, no successor
            # if heu == 0:
            #     continue
            # append parent's successors except this state
            for i in range(heu+1, min(max_heu+1, heu+1+step+1)):
                new_pair_set[state] += successor_set[by_index_state[i]]
            new_pair_set[state].remove(state)
        return new_pair_set

    def generate_one_step_pair_dataset(self, origin_state_pairs):
        successor_set = {}
        by_index_state = {}
        new_pair_set = {}
        max_heu = max(origin_state_pairs.values())

        for state, heu in origin_state_pairs.items():
            # state = self.to_partial_state(state)
            by_index_state[heu] = state
            successor_set[state] = [pair[1] for pair in self.get_successors(state)]

        for heu, state in by_index_state.items():
            # init state, no worse state
            if heu == max_heu:
                continue
            # append parent state
            new_pair_set[state] = [by_index_state[heu + 1]]
            # final state, no successor
            if heu == 0:
                continue
            # append parent's successors except this state
            new_pair_set[state] += successor_set[by_index_state[heu+1]]
            new_pair_set[state].remove(state)
        return new_pair_set

    def goose_state_to_pyperplan(self, state):
        facts = state.split("), (")
        new_facts = []
        for fact in facts:
            fact = fact.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            fact = fact.split(",")
            if fact[0] == "''":
                continue
            fact_name = fact[0].replace("'", "")
            fact_vars = fact[1:]
            new_fact = "(" + fact_name + " "
            for var in fact_vars:
                var = var.replace("'", "").strip()
                new_fact += var + " "
            new_facts.append(new_fact.strip() + ")")

        return frozenset(new_facts)

if __name__ == "__main__":
    domain = "blocks"
    problem = "blocks4-task02.pddl"
    # problem = "p-l2-c10-s1.pddl"
    domain_f = f"/home/ming/PycharmProjects/goose/benchmarks/goose/{domain}/domain.pddl"
    problem_f = f"/home/ming/PycharmProjects/goose/benchmarks/goose/{domain}/train/{problem}"
    solution_f = f"/home/ming/PycharmProjects/goose/benchmarks/goose/{domain}/train_solution/{problem.replace('.pddl', '_1800.out')}"
    problem = STRIPSProblem(domain_f, problem_f, solution_f)
    problem.get_state_heuristic(problem.initial_state)
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
    print(problem.generate_extended_state_dataset(problem.state_to_heuristic))