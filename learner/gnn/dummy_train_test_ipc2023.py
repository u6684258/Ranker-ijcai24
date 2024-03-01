"""
Main script for running GOOSE experiments for ICAPS-24. The experiment pipeline consists of just
1. training
2. testing with search with GPU evaluation
"""
import itertools
import os
import sys

from torch_geometric.data import Data

from util.mdpsim_api import STRIPSProblem
from util.save_load import load_gnn_model_and_setup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import re
import argparse
import numpy as np
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS
from representation import REPRESENTATIONS
from util.scrape_log import scrape_search_log, scrape_train_log, search_finished_correctly

_SEARCH = "gbbfs"

_MODEL_DIR = "./../logs/ranker_gnn_models"
_TRAIN_LOG_DIR = f"./../logs/ranker_train_logs"
_TEST_LOG_DIR = f"./../logs/ranker_test_logs"
os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_TRAIN_LOG_DIR, exist_ok=True)
os.makedirs(_TEST_LOG_DIR, exist_ok=True)

_AUX_DIR = "./../logs/aux"
_PLAN_DIR = "./../logs/plans"
os.makedirs(_AUX_DIR, exist_ok=True)
os.makedirs(_PLAN_DIR, exist_ok=True)

BENCHMARK_DIR = "./../benchmarks/ipc2023-learning-benchmarks"

IPC2023_FAIL_LIMIT = {
    "dummy":1
}

REPEATS = 1

DOWNWARD_GPU_CMD = "./../planners/downward_gpu/fast-downward.py"


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def fd_cmd(df, pf, m, search, timeout=2000):  # 1800s + overhead for timeout
    if search == "gbbfs":
        search = "batch_eager_greedy"
    elif search == "gbfs":
        search = "eager_greedy"
    else:
        raise NotImplementedError

    description = f"fd_{pf.replace('.pddl','').replace('/','-')}_{search}_{os.path.basename(m).replace('.dt', '')}"
    sas_file = f"{_AUX_DIR}/{description}.sas_file"
    plan_file = f"{_PLAN_DIR}/{description}.plan"
    cmd = (
        f"{DOWNWARD_GPU_CMD} --search-time-limit {timeout} --sas-file {sas_file} --plan-file {plan_file} "
        + f'{df} {pf} --search \'{search}([goose(model_path="{m}", domain_file="{df}", instance_file="{pf}")])\''
    )
    cmd = f"export GOOSE={os.getcwd()} && {cmd}"
    return cmd, sas_file


def get_model_desc(rep, domain, L, H, aggr, repeat, model):
    return f"{domain}_{rep}_L{L}_H{H}_{aggr}_r{repeat}_{model}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--aggregation", required=True, choices=["mean", "max"])
    parser.add_argument("-l", "--layers", required=True, type=int, choices=[4, 8])
    parser.add_argument("-r", "--representation", required=True, choices=["ilg", "llg"])
    parser.add_argument("-d", "--domain", required=True, choices=IPC2023_LEARNING_DOMAINS)
    parser.add_argument("-m", "--model", required=True, choices=["gnn", "gnn-rank", "gnn-loss"])
    args = parser.parse_args()

    rep = args.representation
    domain = args.domain
    model_name = args.model
    H = 64
    patience = 10
    aggr = args.aggregation
    L = args.layers
    for num in [10, 20, 30, 40, 50]:
    # num = 50
        df = f"{BENCHMARK_DIR}/{domain}/domain.pddl"  # domain file
        test_dir = f"{BENCHMARK_DIR}/{domain}/testing/easy/n{num}.pddl"


        ###############################################################################################
        for repeat in range(REPEATS):
            ###############################################################################################
            """train"""
            os.system("date")

            desc = get_model_desc(rep, domain, L, H, aggr, repeat, model_name)
            model_file = f"{_MODEL_DIR}/{desc}.dt"

            ###############################################################################################
            """ test """
            model = load_gnn_model_and_setup(model_file, df, test_dir)

            problem = STRIPSProblem(df, [test_dir])
            problem.change_problem(test_dir)
            # state_raw = {"(at n1)"}
            reps = REPRESENTATIONS[rep](domain_pddl=df, problem_pddl=test_dir)
            reps.convert_to_pyg()

            ns = []
            state_raw = problem.initial_state.to_frozen_tuple()
            for i in range(1, num):
                state = reps.str_to_state(state_raw)
                hv = model.h(state)
                ns.append(hv)
                print(f"n{i}: {hv}")
                state_raw.remove(f"(at n{i})")
                state_raw.remove(f"(connected n{i} n{i+1})")
                state_raw.add(f"(at n{i+1})")

            rs = []
            state_raw = problem.initial_state.to_frozen_tuple()
            state_raw.remove(f"(at n1)")
            state_raw.remove(f"(connected n1 r1)")
            state_raw.add(f"(at r1)")
            for i in range(1, num-1):
                state = reps.str_to_state(state_raw)
                hv = model.h(state)
                rs.append(hv)
                print(f"r{i}: {hv}")
                state_raw.remove(f"(at r{i})")
                state_raw.remove(f"(connected n{i} n{i+1})")
                state_raw.remove(f"(connected n{i+1} r{i+1})")
                state_raw.add(f"(at r{i + 1})")
                state_raw.add(f"(connected n{i} r{i})")

            # print(np.array(ns[1:]).reshape([-1,])-np.array(rs).reshape([-1,]))
            print(np.mean(np.array(ns[1:]).reshape([-1,])-np.array(rs).reshape([-1,])))
        ###############################################################################################
