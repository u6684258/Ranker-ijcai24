"""
Main script for running GOOSE experiments for ICAPS-24. The experiment pipeline consists of just
1. training
2. testing with search with GPU evaluation
"""
import re
import itertools
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

from multiprocessing import Pool, Value
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS
from util.scrape_log import scrape_search_log, scrape_train_log, search_finished_correctly

_SEARCH = "gbbfs"
_MODEL_DIR = "./../logs/gnn_models"
_TRAIN_LOG_DIR = "./../logs/train_logs"
_TEST_LOG_DIR = "./../logs/test_logs"
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

DOWNWARD_GPU_CMD = "./../planners/downward_gpu/fast-downward.py"
os.environ['STRIPS_HGN_NEW'] = f'{os.getcwd()}'
os.environ['FD_HGN'] = f'{os.getcwd()}/../planners/FD-Hypernet-master'

success = Value("i", 0)
failed = Value("i", 0)


def fd_cmd(df, pf, m, model_type, search, timeout=1800):  # 1800s + overhead for timeout
    if search == "gbbfs":
        search = "batch_eager_greedy"
    elif search == "gbfs":
        search = "eager_greedy"
    else:
        raise NotImplementedError

    description = f"fd_{pf.replace('.pddl','').replace('/','-')}_{search}_{os.path.basename(m).replace('.dt', '')}"
    sas_file = f"{_AUX_DIR}/{description}.sas_file"
    plan_file = f"{_PLAN_DIR}/{description}.plan"
    if model_type in ["gnn", "gnn-rank", "gnn-loss"]:
        cmd = (
            f"{DOWNWARD_GPU_CMD} --search-time-limit {timeout} --sas-file {sas_file} --plan-file {plan_file} "
            + f'{df} {pf} --search \'{search}([goose(model_path="{m}", domain_file="{df}", instance_file="{pf}")])\''
        )
        cmd = f"export GOOSE={os.getcwd()} && {cmd}"
    elif model_type in ["hgn", "hgn-rank", "hgn-loss"]:
        cmd = f"./../planners/FD-Hypernet-master/fast-downward.py --search-time-limit {timeout} --sas-file {sas_file} --plan-file {plan_file} " + \
              f"{df} {pf} --translate-options --full-encoding --search-options " \
              f"--search \"eager(single(hgn2(network_file={m}," \
              f"domain_file={df}," \
              f"instance_file={pf}," \
              f"type={model_type})))\""
    else:
        print("Invalid model type for FD!")
    return cmd, sas_file


def get_model_desc(rep, domain, L, H, aggr, repeat, model):
    return f"{domain}_{rep}_L{L}_H{H}_{aggr}_r{repeat}_{model}"


def getTime():
    time_stamp = datetime.now()
    return time_stamp.strftime('%H:%M:%S')


def train(args):
    desc = get_model_desc(args.rep, args.domain, args.layers, 64, args.aggregation, args.seed, args.model)
    model_file = f"{_MODEL_DIR}/{desc}.dt"
    train_log_file = f"{_TRAIN_LOG_DIR}/{desc}.log"
    model_type = args.model

    if not os.path.exists(model_file) or not os.path.exists(train_log_file):
        # os.system(f"echo training with {args.domain} {args.rep}, see {train_log_file}")
        print(getTime(), "Training with", args.domain, args.rep, ", see", train_log_file)
        cmd = f"python3 train_rank.py {args.domain} -m {args.model} -r {args.rep} -L {args.layers} -H {64} --aggr {args.aggregation} --patience {10} --save-file {model_file}"
        os.system(f"{cmd} > {train_log_file}")
    else:
        # os.system(f"echo already trained for {args.domain} {args.rep}, see {train_log_file}")
        print(getTime(), "Already trained for ", args.domain, args.rep, ", see", train_log_file)

def evaluate(args):
    global success, failed
    desc = get_model_desc(args.rep, args.domain, args.layers, 64, args.aggregation, args.seed, args.model)
    model_file = f"{_MODEL_DIR}/{desc}.dt"
    df = f"{BENCHMARK_DIR}/{args.domain}/domain.pddl"  # domain file
    test_dir = f"{BENCHMARK_DIR}/{args.domain}/testing"
    test_configs = list(itertools.product(["easy", "medium", "hard"], [f"p0{i}" for i in range(1, 10)] + [f"p{i}" for i in range(10, 31)],))

    multi_re = []
    with Pool(processes=args.p_num) as pool:
        for difficulty, problem_name in test_configs:
            pf = f"{test_dir}/{difficulty}/{problem_name}.pddl"
            if not os.path.exists(pf):
                print(getTime(), pf, "can not be found! Pass!")
                continue

            test_log_file = f"{_TEST_LOG_DIR}/{args.domain}_{difficulty}_{problem_name}_{desc}.log"
            if os.path.exists(test_log_file) and search_finished_correctly(test_log_file):
                print(getTime(), "Already tested for", args.domain, args.rep, ", see", test_log_file)
                with success.get_lock():
                    success.value += 1
                continue

            re_async = pool.apply_async(run_eval, args=(df, pf, args.domain, difficulty, problem_name, model_file, test_log_file, args.timeout))
            multi_re.append(re_async)

        while True:
            # print("not execute num:", results)
            num_not_finish = len([re for re in multi_re if not re.ready()])
            if failed.value >= args.p_num:
                print(getTime(), "Too many failed, stop evaluating more!!!")
                pool.terminate()
                pool.join()
                break

            if num_not_finish == 0:
                print(getTime(), "Finish all tasks.")
                break

            time.sleep(1)

    print(getTime(), args.domain, "coverage is", str(success.value) + ", failed: " + str(failed.value) + ", not execute:", num_not_finish)


def run_eval(df, pf, domain, difficulty, problem_name, model_file, test_log_file, timeout):
    start = datetime.now()
    task_name = domain + "_" + difficulty + "_" + problem_name
    print(getTime(), "Running test", task_name)

    with open(test_log_file, 'w', encoding='utf-8') as log_file:
        try:
            model_type = model_file.split("_")[-1].split(".")[0]
            # os.system(f"{cmd} > {test_log_file}")
            cmd, _ = fd_cmd(df=df, pf=pf, model_type=model_type, m=model_file, search=_SEARCH, timeout=timeout)
            subprocess.run(cmd, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(getTime(), "Child was terminated by signal -" + str(e.returncode), file=sys.stderr)
        except OSError:
            print(getTime(), "OSError:", cmd)

    # os.system(f"echo testing {args.domain} {args.rep}, see {test_log_file}")
    # os.remove(intermediate_file) if os.path.exists(intermediate_file)

    with open(test_log_file, "r", encoding="utf-8") as f:
        read_log = f.read()
        solved = "Solution found." in read_log
        if solved:
            print(getTime(), "Solved", task_name, ", cost time:", (datetime.now() - start).total_seconds())
            with failed.get_lock() and success.get_lock():
                failed.value = 0
                success.value += 1
        else:
            print(getTime(), "Failed", task_name, ", cost time:", (datetime.now() - start).total_seconds())
            with failed.get_lock():
                failed.value += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-d", "--domain", required=True, choices=IPC2023_LEARNING_DOMAINS)
    parser.add_argument("-r", "--rep", default="llg", choices=["ilg", "llg"], help="representation")
    parser.add_argument("-a", "--aggregation", default="mean", choices=["mean", "max"], help="aggregation")
    parser.add_argument("-l", "--layers", type=int, default=4, choices=[4, 8])
    parser.add_argument("-m", "--model", default="gnn", choices=["gnn", "gnn-rank", "gnn-loss", "hgn", "hgn-rank", "hgn-loss"])
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--p_num", type=int, default=2, help="process number")
    parser.add_argument("--timeout", type=int, default=1800, help="timeout seconds")
    args = parser.parse_args()

    train(args)
    if not args.train_only:
        evaluate(args)


if __name__ == "__main__":
    main()