import sys
sys.path.append("..")
import os
import random
import numpy as np
from tqdm import tqdm
# from util.stats import get_stats
from representation import REPRESENTATIONS
from deadend.deadend import deadend_states

_DOWNWARD = "./../planners/downward/fast-downward.py"
_POWERLIFTED = "./../planners/powerlifted/powerlifted.py"


def sample_from_dict(d, sample, seed):
    random.seed(seed)
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))


def get_plan_info(domain_pddl, problem_pddl, plan_file, args):
    states = []
    actions = []

    planner = args.planner

    with open(plan_file, "r") as f:
        for line in f.readlines():
            if ";" in line:
                continue
            actions.append(line.replace("\n", ""))

    state_output_file = repr(hash(repr(args))).replace("-", "n")
    state_output_file += repr(hash(domain_pddl))+repr(hash(problem_pddl))+repr(hash(plan_file))
    aux_file = state_output_file + ".sas"
    state_output_file = state_output_file + ".states"

    cmd = {
        "pwl": f"export PLAN_PATH={plan_file} "
        + f"&& {_POWERLIFTED} -d {domain_pddl} -i {problem_pddl} -s perfect "
        + f"--plan-file {state_output_file}",
        "fd": f"export PLAN_INPUT_PATH={plan_file} "
        + f"&& export STATES_OUTPUT_PATH={state_output_file} "
        + f"&& {_DOWNWARD} --sas-file {aux_file} {domain_pddl} {problem_pddl} "
        + f'--search \'perfect([blind()])\'',  # need filler h
    }[planner]

    # print("generating plan states with:")
    # print(cmd)

    # disgusting method which hopefully makes running in parallel work fine
    assert not os.path.exists(aux_file), aux_file
    assert not os.path.exists(state_output_file), state_output_file
    output = os.popen(cmd).readlines()
    if output == None:
        print("make this variable seen")
    if os.path.exists(aux_file):
        os.remove(aux_file)

    with open(state_output_file, "r") as f:
        for line in f.readlines():
            if ";" in line:
                continue
            line = line.replace("\n", "")
            s = set()
            for fact in line.split():
                if "(" not in fact:
                    lime = f"({fact})"
                else:
                    pred = fact[: fact.index("(")]
                    fact = fact.replace(pred + "(", "").replace(")", "")
                    args = fact.split(",")[:-1]
                    lime = "(" + " ".join([pred] + args) + ")"
                s.add(lime)
            states.append(s)
    os.remove(state_output_file)

    ret = []
    for i, state in enumerate(states):
        if i == len(actions):
            continue  # ignore the goal state, annoying for learning useful schema
        distance_to_goal = len(states) - i - 1
        action = actions[i]
        ret.append((state, action, distance_to_goal))
    return ret


def get_graphs_from_plans(args):
    print("Generating graphs from plans...")
    graphs = []

    representation = args.rep
    domain = args.domain
    domain_pddl = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    tasks_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy"
    plans_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training_plans"

    for plan_file in tqdm(sorted(list(os.listdir(plans_dir)))):
        problem_pddl = f"{tasks_dir}/{plan_file.replace('.plan', '.pddl')}"
        assert os.path.exists(problem_pddl), problem_pddl
        plan_file = f"{plans_dir}/{plan_file}"
        rep = REPRESENTATIONS[representation](domain_pddl, problem_pddl)

        # rep.convert_to_pyg()
        rep.convert_to_coloured_graph()
        plan = get_plan_info(domain_pddl, problem_pddl, plan_file, args)

        for s, action, distance_to_goal in plan:
            s = rep.str_to_state(s)
            graph = rep.state_to_cgraph(s)
            graphs.append((graph, distance_to_goal))

    print("Graphs generated!")
    return graphs


def get_dataset_from_args(args):
    small_train = args.small_train

    dataset = get_graphs_from_plans(args)
    if small_train:
        random.seed(123)
        dataset = random.sample(dataset, k=1000)

    # get_stats(dataset=dataset, desc="Whole dataset")

    graphs = [data[0] for data in dataset]
    y = np.array([data[1] for data in dataset])

    return graphs, y


def get_deadend_dataset_from_args(args):
    rep = args.rep
    planner = args.planner
    small_train = args.small_train

    domain = args.domain
    domain_pddl = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    tasks_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy"

    deadend_data = deadend_states(domain_pddl, tasks_dir)
    unsolvable_states = deadend_data["unsolvable_states"]
    solvable_states = deadend_data["solvable_states"]
    max_solvable_h = deadend_data["max_solvable_h"]
    
    # balance 50-50 pos and neg
    smaller_set_size = min(len(unsolvable_states), len(solvable_states))
    if small_train:
        smaller_set_size = min(smaller_set_size, 1000)
    unsolvable_states = sample_from_dict(unsolvable_states, smaller_set_size, seed=0)
    solvable_states = sample_from_dict(solvable_states, smaller_set_size, seed=0)
    
    # bring (df, pf) to keys for faster generation of graphs below
    unsolvable_states_ret = {}
    for state, (df, pf) in unsolvable_states.items():
        k = (df, pf)
        if k not in unsolvable_states_ret:
            unsolvable_states_ret[k] = []
        unsolvable_states_ret[k].append(state)
    
    solvable_states_ret = {}
    for state, (df, pf) in solvable_states.items():
        k = (df, pf)
        if k not in solvable_states_ret:
            solvable_states_ret[k] = []
        solvable_states_ret[k].append(state)

    unsolvable_states = unsolvable_states_ret
    solvable_states = solvable_states_ret

    dataset = []

    for states_dict, y_deadend in [(unsolvable_states, 1), (solvable_states, 0)]:
        if y_deadend:
            print("generating unsolvable state graphs...")
        else:
            print("generating solvable state graphs...")
        for (df, pf), states in tqdm(states_dict.items()):
            rep = REPRESENTATIONS[args.rep](domain_pddl=df, problem_pddl=pf)
            rep.convert_to_coloured_graph()
            for state_key in states:
                state = set(state_key)
                state = rep.str_to_state(state)
                graph = rep.state_to_cgraph(state)
                dataset.append((graph, y_deadend))

    get_stats(dataset=dataset, desc="Whole dataset")

    graphs = [data[0] for data in dataset]
    y = np.array([data[1] for data in dataset])

    return graphs, y
