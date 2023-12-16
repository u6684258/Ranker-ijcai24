"""
Main script for running GOOSE experiments for ICAPS-24. The experiment pipeline consists of just
1. training
2. testing with search with GPU evaluation
"""
import itertools
import os
import sys

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
    "blocksworld": 15,
    "childsnack": 15,
    "ferry": 15,
    "floortile": 3,
    "miconic": 15,
    "rovers": 15,
    "satellite": 15,
    "sokoban": 15,
    "spanner": 15,
    "transport": 15,
}

REPEATS = 5

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
    parser.add_argument("-m", "--model", required=True, choices=["gnn", "gnn-rank"])
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    rep = args.representation
    domain = args.domain
    model = args.model

    H = 64
    patience = 10
    aggr = args.aggregation
    L = args.layers

    df = f"{BENCHMARK_DIR}/{domain}/domain.pddl"  # domain file
    test_dir = f"{BENCHMARK_DIR}/{domain}/testing"
    test_configs = list(itertools.product(
        ["easy", "medium", "hard"],
        [f"p0{i}" for i in range(1, 10)] + [f"p{i}" for i in range(10, 31)],
    ))

    ###############################################################################################
    for repeat in range(REPEATS):
        ###############################################################################################
        """train"""
        os.system("date")

        desc = get_model_desc(rep, domain, L, H, aggr, repeat, model)
        model_file = f"{_MODEL_DIR}/{desc}.dt"

        train_log_file = f"{_TRAIN_LOG_DIR}/{desc}.log"

        if not os.path.exists(model_file) or not os.path.exists(train_log_file):
            cmd = f"python3 train_rank.py {domain} -m {model} -r {rep} -L {L} -H {H} --aggr {aggr} --patience {patience} --save-file {model_file}"
            os.system(f"echo training with {domain} {rep}, see {train_log_file}")
            os.system(f"{cmd} > {train_log_file}")
        else:
            os.system(f"echo already trained for {domain} {rep}, see {train_log_file}")

        ###############################################################################################
        """ test """
        if not args.train_only:
            failed = 0

            # warmup first
            pf = f"{test_dir}/easy/p01.pddl"
            assert os.path.exists(pf), pf
            cmd, intermediate_file = fd_cmd(df=df, pf=pf, m=model_file, search=_SEARCH, timeout=30)
            os.system("date")
            os.system(f"echo warming up with {domain} {rep} {pf} {model_file}")
            os.popen(cmd).readlines()
            try:
                os.remove(intermediate_file)
            except OSError:
                pass

            # test on problems
            for difficulty, problem_name in test_configs:
                os.system("date")
                pf = f"{test_dir}/{difficulty}/{problem_name}.pddl"
                assert os.path.exists(pf), pf
                test_log_file = f"{_TEST_LOG_DIR}/{domain}_{difficulty}_{problem_name}_{desc}.log"
                finished_correctly = False
                if os.path.exists(test_log_file):
                    finished_correctly = search_finished_correctly(test_log_file)
                if not finished_correctly:
                    cmd, intermediate_file = fd_cmd(df=df, pf=pf, m=model_file, search=_SEARCH)
                    os.system(f"echo testing {domain} {rep}, see {test_log_file}")
                    os.system(f"{cmd} > {test_log_file}")
                    if os.path.exists(intermediate_file):
                        os.remove(intermediate_file)
                else:
                    os.system(f"echo already tested for {domain} {rep}, see {test_log_file}")

                # check if failed or not
                assert os.path.exists(test_log_file)
                log = open(test_log_file, "r").read()
                solved = "Solution found." in log
                if solved:
                    print("solved")
                    failed = 0
                else:
                    print("failed")
                    failed += 1
                if failed >= IPC2023_FAIL_LIMIT[domain]:
                    break
    ###############################################################################################
