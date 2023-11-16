import re
import argparse
import os
from util.search import search_cmd

TIMEOUT = 1800

""" Main search driver. """

PROFILE_CMD_ = "valgrind --tool=callgrind --callgrind-out-file=callgrind.out --dump-instr=yes --collect-jumps=yes"


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def search_cmd(args):
    search_engine = {
        "pwl": pwl_cmd,
        "fd": fd_cmd,
    }[args.planner]

    problem_pddl = args.problem_pddl
    aux_file = args.aux_file
    plan_file = args.plan_file

    description = f"{os.path.basename(problem_pddl)}_{os.path.basename(args.model_path)}"
    description = description.replace(".dt", "").replace(".pddl", "").replace("/", "-")

    if aux_file is None:
        os.makedirs("lifted", exist_ok=True)
        aux_file = f"lifted/{description}.lifted"

    if plan_file is None:
        os.makedirs("plans", exist_ok=True)
        plan_file = f"plans/{description}.plan"

    cmd = search_engine(args, aux_file, plan_file)
    cmd = f"export GOOSE={os.getcwd()} && {cmd}"
    return cmd, aux_file


def pwl_cmd(args, aux_file, plan_file):
    problem_pddl = args.problem_pddl
    domain_pddl = args.domain_pddl
    search = {
        "lazy": "lazy",
        "eager": "eager",
        "astar": "astar",
        "policy": "policy",
    }[args.algorithm]

    cmd = (
        f"./../planners/powerlifted/powerlifted.py "
        f"-d {domain_pddl} "
        f"-i {problem_pddl} "
        f"-m {args.model_path} "
        f"-e gnn "
        f"-s {search} "
        f"--time-limit {TIMEOUT} "
        f"--seed 0 "
        f"--translator-output-file {aux_file} "
        f"--plan-file {plan_file}"
    )
    return cmd


def fd_cmd(args, aux_file, plan_file):
    problem_pddl = args.problem_pddl
    domain_pddl = args.domain_pddl
    search = {
        # "lazy": "batch_lazy_greedy",
        # "eager": "batch_eager_greedy",
        "lazy": "lazy_greedy",
        "eager": "eager_greedy",
        "astar": "astar",
        "policy": "policy",
    }[args.algorithm]

    cmd = (
        f"./../planners/downward/fast-downward.py "
        f"--search-time-limit {TIMEOUT} "
        f"--sas-file {aux_file} "
        f"--plan-file {plan_file} "
        f" {domain_pddl} "
        f" {problem_pddl} "
        f'--search \'{search}([goose(model_path="{args.model_path}", '
        f'model_type="gnn", '
        f'domain_file="{domain_pddl}", '
        f'instance_file="{problem_pddl}"'
        f")])' "
    )

    if search == "astar":
        cmd = cmd.replace("[", "").replace("]", "")

    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_pddl", type=str, help="path to domain pddl file")
    parser.add_argument("problem_pddl", type=str, help="path to problem pddl file")
    parser.add_argument(
        "model_path",
        type=str,
        help="path to saved model weights",
    )
    parser.add_argument(
        "--planner",
        "-p",
        type=str,
        default="pwl",
        choices=["fd", "pwl"],
        help="base c++ planner",
    )
    parser.add_argument(
        "--algorithm",
        "-s",
        type=str,
        default="eager",
        choices=["lazy", "eager", "astar", "policy"],
        help="solving algorithm using the heuristic",
    )
    parser.add_argument("--timeout", "-t", type=int, default=1800, help="timeout in seconds")
    parser.add_argument(
        "--aux-file",
        type=str,
        default=None,
        help="path of auxilary file such as *.sas or *.lifted",
    )
    parser.add_argument("--plan-file", type=str, default=None, help="path of *.plan file")
    args = parser.parse_args()

    cmd, aux_file = search_cmd(args)
    os.system(cmd)
    if os.path.exists(aux_file):
        os.remove(aux_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_pddl", type=str, help="path to domain pddl file")
    parser.add_argument("task_pddl", type=str, help="path to task pddl file")
    parser.add_argument(
        "model_type",
        type=str,
        choices=["gnn", "kernel", "linear-regression-opt", "kernel-opt"],
        help="learning model",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        type=str,
        help="path to saved model weights",
    )
    parser.add_argument(
        "--planner",
        "-p",
        type=str,
        default="fd",
        choices=["fd", "pwl"],
        help="base c++ planner",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        default="gbbfs",
        choices=["gbbfs", "gbfs"],
        help="search algorithm",
    )
    parser.add_argument("--timeout", "-t", type=int, default=600, help="timeout in seconds")
    parser.add_argument(
        "--aux-file",
        type=str,
        default=None,
        help="path of auxilary file such as *.sas or *.lifted",
    )
    parser.add_argument("--plan-file", type=str, default=None, help="path of *.plan file")
    parser.add_argument("--profile", action="store_true", help="profile with valgrind")
    args = parser.parse_args()

    cmd, intermediate_file = search_cmd(
        df=args.domain_pddl,
        pf=args.task_pddl,
        m=args.model_path,
        model_type=args.model_type,
        planner=args.planner,
        search=args.search,
        timeout=args.timeout,
        seed=0,
        aux_file=args.aux_file,
        plan_file=args.plan_file,
        profile=args.profile,
    )

    print(cmd)
    os.system(cmd)
