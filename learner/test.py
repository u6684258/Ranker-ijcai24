import json
import os
import os.path as osp
import subprocess
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from representation import REPRESENTATIONS
from util.metrics import SearchState, SearchMetrics
from util.search import search_cmd

exp_root = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(os.path.join(exp_root, ".."), "benchmarks/goose")
model_root = os.path.join(exp_root, "trained_models_gnn")
val_log_dir = f"{exp_root}/logs/val/{datetime.now().isoformat()}"
val_result_dir = f"{exp_root}/logs/result/{datetime.now().isoformat()}"


def domain_test(domain, test_file, model_file, mode="val"):
    if mode == "test":
        log_dir = os.path.join(val_result_dir, domain)
    else:
        log_dir = os.path.join(val_log_dir, domain)
    result_file = f"{val_result_dir}/{domain}.json"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(val_result_dir).mkdir(parents=True, exist_ok=True)
    df = f"{base_dir}/{domain}/domain.pddl"
    pd = f"{base_dir}/{domain}/{test_file}"
    matrices = []
    test_progress = tqdm(os.listdir(pd), desc=f"Testing on {test_file}")
    for name in test_progress:
        st = time.time()
        pf = f"{pd}/{name}"
        cmd, intermediate_file = search_cmd(
            df=df,
            pf=pf,
            m=f"{model_root}/{model_file}",
            model_type="gnn",
            planner="fd",
            search="gbbfs",
            timeout=600,
            seed=0,
            profile=False,
          )
        val_log_file = f"{log_dir}/{name.replace('.pddl', '')}_{model_file}_cmd.log"
        os.system(f"{cmd} > {val_log_file}")
        et = time.time()
        print(f"true time: {et - st} seconds")
        try:
            os.remove(intermediate_file)
        except OSError:
            pass
        with open(val_log_file, 'r') as stdout_fp:
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
        if 'Search stopped without finding a solution.' in out_text:
            rv = None
            state = SearchState.failed
        elif ("Time limit has been reached" in out_text
              and "Solution found." not in out_text):
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
            print('something wrong!')
        matrix = SearchMetrics(
            nodes_expanded=expansions,
            plan_length=plan_length,
            heuristic_calls=heuristic_calls,
            heuristic_val_for_initial_state=init_h,
            search_time=time_length,
            search_state=state,
        )
        matrices.append(matrix)

    with open(result_file, "w") as f:
        f.write(json.dumps([x._asdict() for x in matrices]))

    return matrices



if __name__ == "__main__":
    domain_test('ferry', 'test', 'test_ranker.dt')
