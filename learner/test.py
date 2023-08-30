import os
import os.path as osp
import subprocess
from enum import Enum
from typing import NamedTuple

from representation import REPRESENTATIONS
from util.search import search_cmd

exp_root = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(os.path.join(exp_root, ".."), "benchmarks/goose")


class SearchState(Enum):
    # Plan found
    success = "success"
    # No plan found
    failed = "failed"
    # Search Timed out
    timed_out = "timed_out"


class SearchMetrics(NamedTuple):
    nodes_expanded: int
    plan_length: int
    heuristic_calls: int
    heuristic_val_for_initial_state: float
    search_time: float
    search_state: SearchState


def domain_test(domain, test_file, model_file):
    rep = "sdg-el"
    val_log_dir = f"{exp_root}/logs/val/{domain}"

    os.makedirs(val_log_dir, exist_ok=True)
    df = f"{base_dir}/{domain}/domain.pddl"
    pd = f"{base_dir}/{domain}/{test_file}"
    for name in os.listdir(pd):
        pf = f"{pd}/{name}"
        cmd, intermediate_file = search_cmd(rep, domain, df, pf, f"{exp_root}/trained_models/{model_file}", "gbbfs", 0)
        val_log_file = f"{val_log_dir}/{name.replace('.pddl', '')}_{model_file}_cmd.log"
        os.system(f"{cmd} > {val_log_file}")
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
            print("FD timeout!")
        elif "search exit code: 0" in out_text:
            state = SearchState.success
            print('success')
            plan_length = out_text.split("KB] Plan length: ")[1].split("step(s).")[0]
            print(f"plan length: {plan_length}")
            expansions = out_text.split("KB] Expanded ")[1].split("state(s).")[0]
            print(f"expansions: {expansions}")
            heuristic_calls = out_text.split("KB] Evaluated ")[1].split("state(s).")[0]
            print(f"heuristic calls: {heuristic_calls}")
            time_length = out_text.split("KB] Search time: ")[1].split("s")[0]
            print(f"time length: {time_length}")
        else:
            state = SearchState.failed
            print('something wrong!')
        # matrix = SearchMetrics(
        #     nodes_expanded=expansions,
        #     plan_length=plan_length,
        #     heuristic_calls=heuristic_calls,
        #     heuristic_val_for_initial_state=init_h,
        #     search_time=time_length,
        #     search_state=state,
        # )
        #


if __name__ == "__main__":
    domain_test('ferry', 'test_small', 'test-ferry.dt')
