import os
from typing import List

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from dataset.goose_domain_info import get_train_solution_goose_instance_files
from representation import REPRESENTATIONS
from util import pyperplan_api

_SAVE_DIR = "../data/graphs_ranker"


def gen_graph_rep(
        representation: str,
        regenerate: bool,
        domain: str,
        step: int,
) -> None:
    """ Generate graph representations from saved optimal plans. """

    # tasks  = get_ipc_domain_problem_files(del_free=False)
    # tasks += get_all_htg_instance_files(split=True)
    tasks = get_train_solution_goose_instance_files()

    new_generated = 0
    pbar = tqdm(tasks)
    for domain_name, domain_pddl, problem_pddl, solution_file in tasks:
        problem_name = os.path.basename(problem_pddl).replace(".pddl", "")
        # if representation in LIFTED_REPRESENTATIONS and domain_name in GROUNDED_DOMAINS:
        #   continue
        pbar.set_description(f"Generating {representation} graphs for {domain_name} {problem_name}")

        # in case we only want to generate graphs for one specific domain
        if domain is not None and domain != domain_name:
            continue

        new_generated += generate_graph_rep_domain(domain_name=domain_name,
                                                   domain_pddl=domain_pddl,
                                                   problem_pddl=problem_pddl,
                                                   solution_file=solution_file,
                                                   representation=representation,
                                                   regenerate=regenerate,
                                                   step=step)
    print(f"newly generated graphs: {new_generated}")
    return


def generate_graph_rep_domain(
        domain_name: str,
        domain_pddl: str,
        problem_pddl: str,
        solution_file,
        representation: str,
        regenerate: bool,
        step: int,
) -> int:
    """ Saves list of torch_geometric.data.Data of graphs and features to file.
        Returns a new graph was generated or not
    """
    save_file = get_data_path(domain_name,
                              domain_pddl,
                              problem_pddl,
                              representation)
    if os.path.exists(save_file):
        if not regenerate:
            return 0
        else:
            os.remove(save_file)  # make a fresh set of data

    graph = generate_graph_from_domain_problem_pddl(domain_name=domain_name,
                                                    domain_pddl=domain_pddl,
                                                    problem_pddl=problem_pddl,
                                                    solution_file=solution_file,
                                                    representation=representation,
                                                    step=step)
    if graph is not None:
        tqdm.write(f'saving data @{save_file}...')
        torch.save(graph, save_file)
        tqdm.write('data saved!')
        return 1
    return 0


def get_data_path(domain_name: str,
                  domain_pddl: str,
                  problem_pddl: str,
                  representation: str) -> str:
    """ Get path to save file of graph training data of given domain. """
    problem_name = os.path.basename(problem_pddl).replace(".pddl", "")
    save_dir = f'{get_data_dir_path(representation)}/{domain_name}'
    save_file = f'{save_dir}/{problem_name}.data'
    os.makedirs(save_dir, exist_ok=True)
    return save_file


def get_data_dir_path(representation: str) -> str:
    save_dir = f'{_SAVE_DIR}/{representation}'
    # os.makedirs(save_dir, exist_ok=True)
    return save_dir


def generate_graph_from_domain_problem_pddl(
        domain_name: str,
        domain_pddl: str,
        problem_pddl: str,
        solution_file: str,
        representation: str,
        step: int,
) -> None:
    """ Generates a list of graphs corresponding to states in the optimal plan """
    ret = []

    states = get_state_data(domain_name, domain_pddl, problem_pddl, solution_file, step)
    if states is None:
        return None

    # see representation package
    rep = REPRESENTATIONS[representation](domain_pddl, problem_pddl)
    rep.convert_to_pyg()

    problem_name = os.path.basename(problem_pddl).replace(".pddl", "")

    for state, coord in states:
        if REPRESENTATIONS[representation].lifted:
            state = rep.str_to_state(state)

        x, edge_index = rep.state_to_tensor(state)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=0,
            domain=domain_name,
            problem=problem_name,
            coord_x=coord[0],
            coord_y=coord[1]
        )
        ret.append(graph_data)
    return ret


def get_state_data(domain_name: str, domain_pddl: str,
                   problem_pddl: str, solution_file: str, step:int):
    """
    state data format:

    <coord 0>,<coord 1>::[<facts>]
    Args:
        domain_name:
        domain_pddl:
        problem_pddl:

    Returns:

    """
    problem_name = os.path.basename(problem_pddl)

    problem_instance = pyperplan_api.STRIPSProblem(domain_pddl, problem_pddl, solution_file)

    data = []
    # lines = problem_instance.generate_one_step_pair_dataset(problem_instance.state_to_heuristic)
    lines = problem_instance.generate_extended_state_dataset(problem_instance.state_to_heuristic, step=step)
    index = 0
    for state_good, state_bads in lines.items():
        data.append((state_good, [index, 0]))
        data.extend([(state_bad, [index, i+1]) for i, state_bad in enumerate(state_bads)])
        index+=1
    return data


def get_graph_data_by_prob(
        representation: str,
        domain: str = "all",
) -> List[List[Data]]:
  """ Load stored generated graphs """

  print("Loading train data...")
  print("NOTE: the data has been precomputed and saved.")
  print("Exec 'python3 scripts_graphs/generate_graphs_gnn.py --regenerate' if representation has been updated!")

  path = get_data_dir_path(representation=representation)
  print(f"Path to data: {path}")

  ret = []

  for data in sorted(list(os.listdir(f"{path}/{domain}"))):
    next_data = torch.load(f'{path}/{domain}/{data}')
    ret.append(next_data)

  print(f"{domain} dataset of {len(ret)} problems loaded!")
  return ret


def get_graph_data(
        representation: str,
        domain: str = "all",
) -> List[Data]:
    """ Load stored generated graphs """

    print("Loading train data...")
    print("NOTE: the data has been precomputed and saved.")
    print("Exec 'python3 scripts_kernel/generate_graphs_kernel.py --regenerate' if representation has been updated!")

    path = get_data_dir_path(representation=representation)
    print(f"Path to data: {path}")

    ret = []

    for data in sorted(list(os.listdir(f"{path}/{domain}"))):
        next_data = torch.load(f'{path}/{domain}/{data}')
        ret += next_data

    print(f"{domain} dataset of size {len(ret)} loaded!")
    return ret