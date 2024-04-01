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
from util.scrape_log import scrape_search_log, scrape_train_log, search_finished_correctly


_TRAIN_LOG_DIR = f"./../logs/astar_train_logs"
_TEST_LOG_DIR = f"./../logs/astar_test_logs"
os.makedirs(_TRAIN_LOG_DIR, exist_ok=True)
os.makedirs(_TEST_LOG_DIR, exist_ok=True)

_AUX_DIR = "./../logs/aux"
_PLAN_DIR = "./../logs/plans"
os.makedirs(_AUX_DIR, exist_ok=True)
os.makedirs(_PLAN_DIR, exist_ok=True)

BENCHMARK_DIR = "./../benchmarks/ipc2023-learning-benchmarks"

IPC2023_FAIL_LIMIT = {
    "blocksworld": 2,
    "childsnack": 2,
    "ferry": 2,
    "floortile": 2,
    "miconic": 2,
    "rovers": 2,
    "satellite": 2,
    "sokoban": 2,
    "spanner": 2,
    "transport": 2,
}

DOWNWARD_GPU_CMD = "./../planners/downward/fast-downward.py"


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def fd_cmd(df, pf, timeout=2000):  # 1800s + overhead for timeout

    description = f"fd_{pf.replace('.pddl','').replace('/','-')}_astar"
    sas_file = f"{_AUX_DIR}/{description}.sas_file"
    plan_file = f"{_PLAN_DIR}/{description}.plan"
    cmd = (
        f"{DOWNWARD_GPU_CMD} --search-time-limit {timeout} --sas-file {sas_file} --plan-file {plan_file} "
        + f'{df} {pf} --search \'astar(lmcut())\''
    )
    cmd = f"export GOOSE={os.getcwd()} && {cmd}"
    return cmd, sas_file


def get_model_desc(rep, domain, L, H, aggr, repeat, model):
    return f"{domain}_{rep}_L{L}_H{H}_{aggr}_r{repeat}_{model}"


if __name__ == "__main__":

    for domain in IPC2023_FAIL_LIMIT.keys():

        df = f"{BENCHMARK_DIR}/{domain}/domain.pddl"  # domain file
        test_dir = f"{BENCHMARK_DIR}/{domain}/testing"
        test_configs = list(itertools.product(
            ["easy"],
            [f"p0{i}" for i in range(1, 10)] + [f"p{i}" for i in range(10, 20)]
        ))

        ###############################################################################################
        """ test """
        failed = 0

        # warmup first
        pf = f"{test_dir}/easy/p01.pddl"
        assert os.path.exists(pf), pf
        cmd, intermediate_file = fd_cmd(df=df, pf=pf, timeout=30)
        os.system("date")
        os.system(f"echo warming up with {domain} {pf} astar")
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
            test_log_file = f"{_TEST_LOG_DIR}/{domain}_{difficulty}_{problem_name}_astar.log"
            finished_correctly = False
            if os.path.exists(test_log_file):
                finished_correctly = search_finished_correctly(test_log_file)
            if not finished_correctly:
                cmd, intermediate_file = fd_cmd(df=df, pf=pf)
                os.system(f"echo testing {domain}, see {test_log_file}")
                os.system(f"{cmd} > {test_log_file}")
                if os.path.exists(intermediate_file):
                    os.remove(intermediate_file)
            else:
                os.system(f"echo already tested for {domain}, see {test_log_file}")

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
