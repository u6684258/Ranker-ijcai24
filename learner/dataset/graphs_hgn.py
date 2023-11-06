import json
import os
from typing import List, Dict, Tuple

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from dataset.goose_domain_info import get_train_solution_goose_instance_files
from representation import REPRESENTATIONS, HypergraphsTuple
from representation.hypergraph_nets.delete_relaxation import DeleteRelaxationHypergraphView
from representation.hypergraph_nets.features.global_features import EmptyGlobalFeatureMapper
from representation.hypergraph_nets.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from representation.hypergraph_nets.features.node_features import PropositionInStateAndGoal
from representation.hypergraph_nets.hypergraph_nets_adaptor import hypergraph_view_to_hypergraphs_tuple
from representation.hypergraph_nets.hypergraph_view import HypergraphView
from util import pyperplan_api
from util.pyperplan_api import STRIPSProblem, State

_SAVE_DIR = "../data/graphs_hgn"


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

    print("Gathering problems...")
    problems = {}
    domain_to_max = {}
    for domain_name, domain_pddl, problem_pddl, solution_file in tasks:
        if domain is not None and domain != domain_name:
            continue
        if domain_name not in problems.keys():
            problems[domain_name] = []
            print(domain)
        problem = STRIPSProblem(domain_pddl, problem_pddl, solution_file)
        problems[domain_name].append([max(
            len(action.add_effects)
            for action in problem.actions
        ), max(
            len(action.preconditions)
            for action in problem.actions
        ), problem.number_of_propositions,
            problem.number_of_actions,
        ])
    print("Computing max_receivers and max_senders...")
    for domain, vars in problems.items():
        domain_to_max[domain] = [max(var[0] for var in vars), max(var[1] for var in vars), max(var[2] for var in vars), max(var[3] for var in vars)]

    json.dump(domain_to_max, open(f"{_SAVE_DIR}/hyperparameters.json", "w"))
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
                                                   step=step,
                                                   max_receivers=domain_to_max[domain_name][0],
                                                   max_senders=domain_to_max[domain_name][1],
                                                   )
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
        max_receivers,
        max_senders,
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
                                                    step=step,
                                                    max_receivers=max_receivers,
                                                    max_senders=max_senders,
                                                    )
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


def create_input_and_target_hypergraphs_tuple(
        state: State, hypergraph: HypergraphView,
        max_receivers, max_senders, coords, heu_value
) -> HypergraphsTuple:


    # The input HypergraphsTuple with its node and hyperedge features.
    global_features = hypergraph.global_features(
        global_feature_mapper=EmptyGlobalFeatureMapper()
    )
    global_features = (
        torch.tensor(global_features, dtype=torch.float32).reshape(1, -1)
    )

    input_h_tuple = hypergraph_view_to_hypergraphs_tuple(
        hypergraph=hypergraph,
        receiver_k=max_receivers,
        sender_k=max_senders,
        # Map the nodes to their features
        node_features=torch.tensor(
            hypergraph.node_features(
                PropositionInStateAndGoal(
                    state, hypergraph.problem.goals
                )
            ),
            dtype=torch.float32,
        ),
        # Map the hyperedges to their features
        edge_features=torch.tensor(
            hypergraph.hyperedge_features(
                ComplexHyperedgeFeatureMapper()
            ),
            dtype=torch.float32,
        ),
        # Map the hypergraph to its global features
        y_value=torch.tensor(heu_value, dtype=torch.float32).reshape(-1,),
        global_features=global_features,
        x_coord=torch.tensor(coords[0], dtype=torch.float32).reshape(-1,),
        y_coord=torch.tensor(coords[1], dtype=torch.float32).reshape(-1,),
    )

    return input_h_tuple


def generate_graph_from_domain_problem_pddl(
        domain_name: str,
        domain_pddl: str,
        problem_pddl: str,
        solution_file: str,
        representation: str,
        max_receivers,
        max_senders,
        step: int,
) -> list[HypergraphsTuple] | None:
    """ Generates a list of graphs corresponding to states in the optimal plan """
    ret = []

    if representation == "hgn":
        states, problem_instance = get_plan_data(domain_name, domain_pddl, problem_pddl, solution_file)
    else:
        states, problem_instance = get_state_data(domain_name, domain_pddl, problem_pddl, solution_file, step)
    if states is None:
        return None

    # HGN graph generation
    problem_to_delete_relaxation_hypergraph = DeleteRelaxationHypergraphView(problem_instance)

    if representation == "hgn":
        kfold_hypergraphs_tuples = [
            create_input_and_target_hypergraphs_tuple(
                training_pair[0],
                problem_to_delete_relaxation_hypergraph,
                max_receivers,
                max_senders,
                coords=[0,0],
                heu_value=training_pair[1]
            )
            for training_pair in states
        ]

    else:
        kfold_hypergraphs_tuples = [
            create_input_and_target_hypergraphs_tuple(
                training_pair[0],
                problem_to_delete_relaxation_hypergraph,
                max_receivers,
                max_senders,
                coords=training_pair[1],
                heu_value=0,
            )
            for training_pair in states
        ]

    problem_name = os.path.basename(problem_pddl).replace(".pddl", "")

    return kfold_hypergraphs_tuples


def get_state_data(domain_name: str, domain_pddl: str,
                   problem_pddl: str, solution_file: str, step: int):
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
        data.extend([(state_bad, [index, i + 1]) for i, state_bad in enumerate(state_bads)])
        index += 1
    return data, problem_instance


def get_plan_data(domain_name: str, domain_pddl: str,
                   problem_pddl: str, solution_file: str):
    """
    state data format:

    [<facts>], Heuristic Value
    Args:
        domain_name:
        domain_pddl:
        problem_pddl:

    Returns:

    """
    problem_instance = pyperplan_api.STRIPSProblem(domain_pddl, problem_pddl, solution_file)

    lines = [(k, v) for k, v in problem_instance.state_to_heuristic.items()]
    return lines, problem_instance

def get_graph_data_by_prob(
        representation: str,
        domain: str = "all",
) -> List[List[HypergraphsTuple]]:
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

    hyps = json.load(open(f"{_SAVE_DIR}/hyperparameters.json", "r"))
    return ret, hyps


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
