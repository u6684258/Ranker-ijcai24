import os

import test
from util.pyperplan_api import STRIPSProblem

exp_root = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(os.path.join(exp_root, ".."), "benchmarks/goose")


def extract_from_fd_log(benchmark_path, domain, folder, file, model):
    pyperplan_problem = STRIPSProblem(f"{benchmark_path}/domain.pddl", f"{probs_path}/{file}")
    _, val_log_file = test.single_test(domain, folder, file, model)

    states = []
    heus = []
    with open(val_log_file, "r") as f:
        while line := f.readline():
            if "goose state: " in line:
                stateh = line.split("goose state: ")[1].split("h: ")
                state = pyperplan_problem.goose_state_to_pyperplan(stateh[0])
                states.append(state)
                true_heu = pyperplan_problem.get_state_heuristic(state)
                heus.append((int(stateh[1]), true_heu))

    return states, heus
def extract_from_folder(domain, folder, model):
    benchmark_path = os.path.join(base_dir, domain)
    probs_path = os.path.join(benchmark_path, folder)
    for file in os.listdir(probs_path):
        extract_from_fd_log(benchmark_path, domain, folder, file, model)


def get_stats(states, heus):
    accuracy = 0
    for i, state in enumerate(states):
        heu, heu_t = heus[i]


if __name__ == "__main__":
    domain = "blocks"
    folder = "train"
    file = "blocks4-task01.pddl"
    model = "rank-blocks-L4-coord.dt"
    benchmark_path = os.path.join(base_dir, domain)
    probs_path = os.path.join(benchmark_path, folder)
    extract_from_fd_log(benchmark_path, domain, folder, file, model)