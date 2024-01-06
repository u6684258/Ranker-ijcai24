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
from ranker.rank_dataset import BatchSampler, ByProblemDataset
from util import mdpsim_api
from util.mdpsim_api import State

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random
from torch.utils.data import DataLoader

_DOWNWARD = "./../planners/downward/fast-downward.py"
_POWERLIFTED = "./../planners/powerlifted/powerlifted.py"
DATA_DIR = "./../data/ipc23/hgn-rank"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_plan_info(problems, problem_pddl, plan_file, args):
    problems.change_problem(problem_pddl, plan_file)

    lines = [(k, v) for k, v in problems.generate_extended_state_dataset(problems.state_to_heuristic).items()]

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
                    state.to_frozen_tuple(),
                    set([f"({i.unique_ident})" for i in hypergraph.problem.problem_meta.goal_props])
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
        y_value=torch.tensor(heu_value, dtype=torch.float32).reshape(-1, ),
        global_features=global_features,
        x_coord=torch.tensor(coords[0], dtype=torch.float32).reshape(-1, ),
        y_coord=torch.tensor(coords[1], dtype=torch.float32).reshape(-1, ),
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
        for i, (state, pairs) in enumerate(plan):
            for j, pair in enumerate([state] + pairs):
                graph = create_input_and_target_hypergraphs_tuple(
                    state,
                    problem_to_delete_relaxation_hypergraph,
                    problems.max_receivers,
                    problems.max_senders,
                    coords=[i, j],
                    heu_value=0
                )
                problem_set.append(graph)
        graphs.append(problem_set)

    print("Graphs generated!")
    return graphs, problems.max_receivers, problems.max_senders


def get_loaders_from_args_hgn_rank(args):
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

    val_interval = round(1 / args.val_ratio)
    train_valid_count = 0
    val_valid_count = 0
    trainset = []
    valset = []
    for i, problem in enumerate(dataset):
        feature_set = []
        if i % val_interval == 0:
            for data in problem:
                feature_set.append(data.replace(p_idx=val_valid_count))
            valset.extend(feature_set)
            val_valid_count += 1
        else:
            for data in problem:
                feature_set.append(data.replace(p_idx=train_valid_count))
            trainset.extend(feature_set)
            train_valid_count += 1

    train_batch_sampler = BatchSampler(ByProblemDataset(trainset,
                                                        train_valid_count).per_class_sample_indices(),
                                       batch_size=batch_size)
    train_loader = DataLoader(
        trainset,
        pin_memory=True,
        batch_sampler=train_batch_sampler,
        collate_fn=merge_hypergraphs_tuple
    )
    val_batch_sampler = BatchSampler(ByProblemDataset(valset,
                                                      val_valid_count).per_class_sample_indices(),
                                     batch_size=batch_size)

    val_loader = DataLoader(
        valset,
        pin_memory=True,
        batch_sampler=val_batch_sampler,
        collate_fn=merge_hypergraphs_tuple
    )
    # get_stats(dataset=list(itertools.chain.from_iterable(dataset)), desc="Whole dataset")
    # get_stats(dataset=trainset, desc="Train set")
    # get_stats(dataset=valset, desc="Val set")
    print("train size:", len(trainset))
    print("validation size:", len(valset))

    return train_loader, val_loader, m_re, m_se
