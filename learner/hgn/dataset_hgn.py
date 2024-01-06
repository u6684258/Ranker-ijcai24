import os
import sys
import itertools
from pathlib import Path

import torch

from hgn.hypergraph_nets.delete_relaxation import DeleteRelaxationHypergraphView
from hgn.hypergraph_nets.features.global_features import EmptyGlobalFeatureMapper
from hgn.hypergraph_nets.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from hgn.hypergraph_nets.features.node_features import PropositionInStateAndGoal
from hgn.hypergraph_nets.hypergraph_nets_adaptor import hypergraph_view_to_hypergraphs_tuple, merge_hypergraphs_tuple
from hgn.hypergraph_nets.hypergraph_view import HypergraphView
from hgn.hypergraph_nets.hypergraphs import HypergraphsTuple
from util import mdpsim_api
from util.mdpsim_api import State

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

_DOWNWARD = "./../planners/downward/fast-downward.py"
_POWERLIFTED = "./../planners/powerlifted/powerlifted.py"
DATA_DIR = "./../data/ipc23/hgn"


def get_plan_info(problems, problem_pddl, plan_file, args):

    problems.change_problem(problem_pddl, plan_file)

    lines = [(k, v) for k, v in problems.state_to_heuristic.items()]

    return lines


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
                    state.to_frozen_tuple(), set([f"({i.unique_ident})" for i in hypergraph.problem.problem_meta.goal_props])
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
        p_idx=0,
        y_value=torch.tensor(heu_value, dtype=torch.float32).reshape(-1,),
        global_features=global_features,
        x_coord=torch.tensor(coords[0], dtype=torch.float32).reshape(-1,),
        y_coord=torch.tensor(coords[1], dtype=torch.float32).reshape(-1,),
    )

    return input_h_tuple
def get_tensor_graphs_from_plans(args):
    print("Generating graphs from plans...")
    graphs = []

    domain_pddl = args.domain_pddl
    tasks_dir = args.tasks_dir
    plans_dir = args.plans_dir
    problem_pddls = []
    for plan_file in sorted(list(os.listdir(plans_dir))):
        problem_pddl = f"{tasks_dir}/{plan_file.replace('.plan', '.pddl')}"
        assert os.path.exists(problem_pddl), problem_pddl
        problem_pddls.append(problem_pddl)

    problems = mdpsim_api.STRIPSProblem(domain_pddl, problem_pddls)

    for problem_pddl in problem_pddls:
        problem_set = []
        plan_file = f"{plans_dir}/{problem_pddl.split('/')[-1].replace('.pddl', '.plan')}"
        plan = get_plan_info(problems, problem_pddl, plan_file, args)
        problem_to_delete_relaxation_hypergraph = DeleteRelaxationHypergraphView(problems)
        for state, schema_cnt in plan:
            graph = create_input_and_target_hypergraphs_tuple(
                state,
                problem_to_delete_relaxation_hypergraph,
                problems.max_receivers,
                problems.max_senders,
                coords=[0,0],
                heu_value=schema_cnt
            )
            problem_set.append(graph)
        graphs.append(problem_set)

    print("Graphs generated!")
    return graphs, problems.max_receivers, problems.max_senders


def get_loaders_from_args_hgn(args):
    batch_size = args.batch_size
    small_train = args.small_train
    data_dir = Path(f"{DATA_DIR}/{args.domain_pddl.split('/')[-2]}.data")
    if data_dir.is_file():
        print(f"Loading graphs from {data_dir}...")
        dataset, m_re, m_se = torch.load(data_dir)
    else:
        dataset, m_re, m_se = get_tensor_graphs_from_plans(args)
        torch.save((dataset, m_re, m_se), data_dir)
    if small_train:
        random.seed(123)
        dataset = random.sample(dataset, k=10)

    trainset, valset = train_test_split(dataset, test_size=0.10, random_state=4550)
    trainset = list(itertools.chain.from_iterable(trainset))
    valset = list(itertools.chain.from_iterable(valset))
    # get_stats(dataset=list(itertools.chain.from_iterable(dataset)), desc="Whole dataset")
    # get_stats(dataset=trainset, desc="Train set")
    # get_stats(dataset=valset, desc="Val set")
    print("train size:", len(trainset))
    print("validation size:", len(valset))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=merge_hypergraphs_tuple,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=merge_hypergraphs_tuple,
    )

    return train_loader, val_loader, m_re, m_se
