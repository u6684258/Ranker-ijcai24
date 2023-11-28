import re
import argparse
import os
from util.save_load import load_kernel_model_and_setup

TIMEOUT = 1800000

""" Main search driver. """

_DOWNWARD = "./../planners/downward/fast-downward.py"
_PROFILE_CMD = "valgrind --tool=callgrind --callgrind-out-file=callgrind.out --dump-instr=yes --collect-jumps=yes"


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def get_profile_cmd(cmd, model_data, graph_data):
    import shutil

    if model_data is not None:
        shutil.copyfile(model_data, model_data + "-copy")
    if graph_data is not None:
        shutil.copyfile(graph_data, graph_data + "-copy")
    print("Running the original command to get individual commands for profiling...")
    output = os.popen(f"export GOOSE={os.getcwd()} && {cmd}").readlines()
    translator_cmd = ""
    search_cmd = ""
    for line in output:
        if "INFO     translator command line string:" in line:
            translator_cmd = line.replace("INFO     translator command line string:", "").replace(
                "\n", ""
            )
            continue
        if "INFO     search command line string:" in line:
            search_cmd = line.replace("INFO     search command line string:", "")
            continue
    if model_data is not None:
        shutil.move(model_data + "-copy", model_data)
    if graph_data is not None:
        shutil.move(graph_data + "-copy", graph_data)
    cmd = f"{translator_cmd} && {_PROFILE_CMD} {search_cmd}"
    print("Original command completed.")
    return cmd


def search_cmd(args):
    get_search_cmd = {
        "fd": fd_cmd,
    }[args.planner]

    aux_file = args.aux_file
    plan_file = args.plan_file

    description = repr(hash(repr(args))).replace("-", "n")

    if aux_file is None:
        os.makedirs("aux", exist_ok=True)
        aux_file = f"aux/{description}.aux"

    if plan_file is None:
        os.makedirs("plans", exist_ok=True)
        plan_file = f"plans/{description}.plan"

    cmd = get_search_cmd(args, aux_file, plan_file)
    cmd = f"export GOOSE={os.getcwd()} && {cmd}"
    return cmd, aux_file


def fd_cmd(args, aux_file, plan_file):
    m = args.model_path
    df = args.domain_pddl
    pf = args.problem_pddl
    search = {
        "lazy": "lazy_greedy",
        "eager": "eager_greedy",
    }[args.algorithm]

    model = load_kernel_model_and_setup(m, df, pf)
    model_type = {
        "linear-svr": "linear_model",
        "ridge": "linear_model",
        "lasso": "linear_model",
        "rbf-svr": "kernel_model",
        "quadratic-svr": "kernel_model",
        "cubic-svr": "kernel_model",
        "mlp": "kernel_model",
        "blr": "bayes_model",
        "gp": "linear_model",  # we assume dot product kernel
    }[model.model_name]

    if model_type == "linear_model":
        model.write_model_data()
        model.write_representation_to_file()
        model_data = model.get_model_data_path()
        graph_data = model.get_graph_file_path()

        fd_h = f'{model_type}(model_data="{model_data}", graph_data="{graph_data}")'
    elif model_type in {"kernel_model", "bayes_model"}:
        model_data = None
        graph_data = None

        fd_h = f'{model_type}(model_data="{m}", domain_file="{df}", instance_file="{pf}")'
    else:
        raise ValueError(model_type)

    cmd = f"{_DOWNWARD} --search-time-limit {args.timeout} --sas-file {aux_file} --plan-file {plan_file} {df} {pf} --search '{search}([{fd_h}])'"

    if args.profile:
        cmd = get_profile_cmd(cmd, model_data, graph_data)

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
        default="fd",
        choices=["fd"],
        help="base c++ planner",
    )
    parser.add_argument(
        "--algorithm",
        "-s",
        type=str,
        default="eager",
        choices=["eager", "lazy"],
        help="solving algorithm using the heuristic",
    )
    parser.add_argument("--timeout", "-t", type=int, default=TIMEOUT, help="timeout in seconds")
    parser.add_argument("--profile", action="store_true", help="profile with valgrind")
    parser.add_argument(
        "--aux-file",
        type=str,
        default=None,
        help="path of auxilary file such as *.sas or *.lifted",
    )
    parser.add_argument(
        "--plan-file",
        type=str,
        default=None,
        help="path of *.plan file",
    )
    args = parser.parse_args()

    cmd, aux_file = search_cmd(args)
    print(cmd)
    os.system(cmd)
    if os.path.exists(aux_file):
        os.remove(aux_file)
