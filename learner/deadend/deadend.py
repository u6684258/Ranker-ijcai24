import os
from typing import Dict, List, Tuple
from dlplan.state_space import generate_state_space, GeneratorExitCode
from dlplan.core import SyntacticElementFactory
from tqdm import tqdm
from planning import get_planning_problem


def dlplan_prop_repr_to_pyperplan_prop_repr(prop: str):
    """from pred(arg1,...,argn) to (pred arg1 ... argn)"""
    pred = prop.split("(")[0]
    args = prop[prop.index("(") + 1 : prop.index(")")].split(",")
    prop = "(" + " ".join([pred] + args) + ")"
    return prop


def deadend_states(domain_pddl, tasks_dir, max_states_expand=10000) -> Tuple[Dict, Dict]:
    problem_files = sorted([f"{tasks_dir}/{f}" for f in os.listdir(tasks_dir)])

    df = domain_pddl
    pf = problem_files[0]

    # dlplan factory object
    state_space = generate_state_space(df, pf, index=0, max_num_states=1).state_space
    instance_info = state_space.get_instance_info()
    vocabulary_info = instance_info.get_vocabulary_info()
    # factory = SyntacticElementFactory(vocabulary_info)  # not needed

    fd_problem = get_planning_problem(df, pf)
    fd_predicates = set(p.name for p in fd_problem.predicates)

    unsolvable_states = {}
    solvable_states = {}
    max_h = 0

    for pf in tqdm(problem_files):
        generator = generate_state_space(
            domain_file=df,
            instance_file=pf,
            vocabulary_info=vocabulary_info,
            index=0,
            max_num_states=max_states_expand,
        )

        if generator.exit_code != GeneratorExitCode.COMPLETE:
            continue
        else:
            tqdm.write(f"  collected state space training data from {pf}")
        state_space = generator.state_space

        ### dlplan static atoms include objects and goal versions of predicates so we prune them
        ## static atoms are added in representation.base_class
        # instance_info = state_space.get_instance_info()
        # static_atoms = set()
        # for prop in instance_info.get_static_atoms():
        #     pred = prop.get_name().split("(")[0]
        #     if pred not in fd_predicates:
        #         continue
        #     static_atoms.add(dlplan_prop_repr_to_pyperplan_prop_repr(prop.get_name()))

        # collect distance for all states; a dead end does not have an entry in goal_distances
        instance_info = state_space.get_instance_info()
        goal_distances = state_space.compute_goal_distances()
        max_h = max(max_h, max(goal_distances.values()))

        for sid, state in state_space.get_states().items():
            state = repr(state)
            i = state.index("{")
            j = state.index("}")
            state_str = state[i + 1 : j]
            state_str = state_str.split(", ")
            state = set(dlplan_prop_repr_to_pyperplan_prop_repr(prop) for prop in state_str)
            # state = state.union(static_atoms)  # fd does not see static atoms
            state = tuple(sorted(state))
            if sid not in goal_distances and state not in unsolvable_states:
                unsolvable_states[state] = (df, pf)
            elif sid in goal_distances and state not in solvable_states:
                solvable_states[state] = (df, pf)

    return {
        "unsolvable_states": unsolvable_states,
        "solvable_states": solvable_states,
        "max_solvable_h": max_h,
    }
