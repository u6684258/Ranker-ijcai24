import json
import multiprocessing
import os
import os.path as osp
import signal
import subprocess
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import psutil
from tqdm import tqdm

from representation import REPRESENTATIONS
from util.metrics import SearchState, SearchMetrics
from util.search import search_cmd, fd_general_cmd

exp_root = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(os.path.join(exp_root, ".."), "benchmarks/goose")
model_root = os.path.join(exp_root, "trained_models_gnn")


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)

def work(cmd, log_file, timeout):
    timeouted = False
    wrong = False
    st = time.time()
    with open(log_file, 'w') as out_fp:
        try:
            print(cmd)
            rv = subprocess.Popen(cmd,
                                  cwd=exp_root,
                                  stdout=out_fp,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True,
                                  )
            rv.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeouted = True
            kill_child_processes(rv.pid)
        except subprocess.CalledProcessError:
            wrong = True
            for line in rv.stdout:
                print(line)
            kill_child_processes(rv.pid)

    et = time.time()

    print(f"true time: {et - st} seconds")

    with open(log_file, 'r') as stdout_fp:
        out_text = stdout_fp.read()
    state = None
    expansions = -1
    heuristic_calls = -1
    plan_length = -1
    time_length = -1
    init_h = -1
    # Check whether search was successful or not. If it was not, return None;
    # if it was, return plan (as list of string-formatted action names, minus
    # parens).
    if not timeouted and 'Search stopped without finding a solution.' in out_text:
        rv = None
        state = SearchState.failed
    elif ("Time limit has been reached" in out_text
          and "Solution found." not in out_text) or timeouted:
        # timeout
        state = SearchState.timed_out
        # print(out_text)
        print("FD timeout!")
        print(f"plan length: {plan_length}")
        expansions = out_text.split("[expansions: ")[-1].split(", ")[0]
        print(f"expansions: {expansions}")
        heuristic_calls = out_text.split(", evaluations: ")[-1].split(", ")[0]
        print(f"heuristic calls: {heuristic_calls}")
        time_length = 600
        print(f"time length: {time_length}")

    elif wrong:
        state = SearchState.failed
        print('something wrong!')


    elif "search exit code: 0" in out_text:
        state = SearchState.success
        print('success')
        # print(out_text)
        plan_length = out_text.split("KB] Plan length: ")[1].split("step(s).")[0]
        print(f"plan length: {plan_length}")
        expansions = out_text.split("KB] Expanded ")[1].split("state(s).")[0]
        print(f"expansions: {expansions}")
        heuristic_calls = out_text.split("KB] Evaluated ")[1].split("state(s).")[0]
        print(f"heuristic calls: {heuristic_calls}")
        time_length = out_text.split("] Search time: ")[1].split("s")[0]
        print(f"time length: {time_length}")
    else:
        state = SearchState.failed
        print("Some uncatched error is happening!")
    matrix = SearchMetrics(
        problem=cmd[8].split("/")[-1],
        nodes_expanded=expansions,
        plan_length=plan_length,
        heuristic_calls=heuristic_calls,
        heuristic_val_for_initial_state=init_h,
        search_time=time_length,
        search_state=state,
    )
    return matrix


def domain_test(domain, test_file, model_file, mode="val", timeout=600, log_root=exp_root):
    val_log_dir = f"{log_root}/val/{model_file}-{datetime.now().isoformat()}"
    val_result_dir = f"{log_root}/result/{model_file}-{datetime.now().isoformat()}"
    if mode == "test":
        log_dir = os.path.join(val_result_dir, domain)
    else:
        log_dir = os.path.join(val_log_dir, domain)
    result_file = f"{log_dir}/{domain}.json"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    df = f"{base_dir}/{domain}/domain.pddl"
    pd = f"{base_dir}/{domain}/{test_file}"
    # test_progress = tqdm(os.listdir(pd), desc=f"Testing on {test_file}")

    jobs = []
    for name in os.listdir(pd):
        pf = f"{pd}/{name}"
        cmd = fd_general_cmd(df, pf, "/tmp/result.txt", search="hff")

        val_log_file = f"{log_dir}/{name.replace('.pddl', '')}_{model_file}_out.log"
        # val_err_file = f"{log_dir}/{name.replace('.pddl', '')}_{model_file}_err.log
        jobs.append((cmd, val_log_file, timeout))

    count = 1
    pool = multiprocessing.Pool(processes=count)
    matrices = []
    r = pool.starmap_async(work, jobs, callback=matrices.append, error_callback=lambda x: print(x))
    r.wait()
    # print(len(matrices[0]))
    with open(result_file, "w") as f:
        f.write(json.dumps([x._asdict() for x in matrices[0]]))

    return matrices[0]


if __name__ == "__main__":
    domain_test('ferry', 'test', 'hff', 'test')
    domain_test('sokoban', 'test', 'hff', 'test')
    domain_test('visitsome', 'test', 'hff', 'test')
    domain_test('visitall', 'test', 'hff', 'test')
    domain_test('gripper', 'test', 'hff', 'test')
    domain_test('spanner', 'test', 'hff', 'test')
    domain_test('n-puzzle', 'test', 'hff', 'test')
    domain_test('blocks', 'test', 'hff', 'test')
    # single_test("ferry", "train", "p-l2-c10-s1.pddl", 'rank-ferry-L4-coord.dt')
