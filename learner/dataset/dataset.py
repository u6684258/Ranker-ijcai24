import itertools
import os
import sys
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util.stats import get_stats
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from dataset.graphs import get_graph_data
from representation.node_features import add_features
from util.transform import extract_testset_ipc, preprocess_data, sample_strategy


def get_loaders_from_args(args):
    model_name = args.model
    batch_size = args.batch_size
    domain = args.domain
    rep = args.rep
    max_nodes = args.max_nodes
    cutoff = args.cutoff
    small_train = args.small_train
    strategy = args.strategy
    num_workers = 0
    pin_memory = True

    dataset = get_graph_data(domain=domain, representation=rep)
    dataset = preprocess_data(model_name, data_list=dataset, c_hi=cutoff, n_hi=max_nodes, small_train=small_train)
    dataset = add_features(dataset, args)
    get_stats(dataset=dataset, desc="Whole dataset")

    trainset, valset = train_test_split(dataset, test_size=0.15, random_state=4550)

    trainset = sample_strategy(data_list=trainset, strategy=strategy)
    get_stats(dataset=trainset, desc="Train set")
    get_stats(dataset=valset, desc="Val set")
    print("train size:", len(trainset))
    print("validation size:", len(valset))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                              num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                            num_workers=num_workers)

    return train_loader, val_loader


def get_paired_dataloaders_from_args(args):
    model_name = args.model
    batch_size = args.batch_size
    domain = args.domain
    rep = args.rep
    max_nodes = args.max_nodes
    cutoff = args.cutoff
    small_train = args.small_train
    strategy = args.strategy
    num_workers = 0
    pin_memory = True

    dataset: List[List[Data]] = get_graph_data(domain=domain, representation=rep, paired=True)
    new_dataset = []
    for datalist in dataset:
        new_datalist = preprocess_data(model_name, data_list=datalist, c_hi=cutoff, n_hi=max_nodes,
                                       small_train=small_train)
        new_datalist = add_features(new_datalist, args)
        new_dataset += itertools.combinations(new_datalist, 2)
        # new_dataset += [(i, j) for i, j in zip(new_datalist, new_datalist[1:])]
        get_stats(dataset=new_datalist, desc="Whole dataset")

    new_dataset = convert_pair_to_data(new_dataset)

    trainset, valset = train_test_split(new_dataset, test_size=0.15, random_state=4550)

    trainset = sample_strategy(data_list=trainset, strategy=strategy)
    get_stats(dataset=trainset, desc="Train set")
    get_stats(dataset=valset, desc="Val set")
    print("train size:", len(trainset))
    print("validation size:", len(valset))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                              num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                            num_workers=num_workers)

    return train_loader, val_loader


def convert_pair_to_data(dataset: List[Tuple[Data, Data]]) -> List[Data]:
    new_dataset = []
    for data_l, data_r in dataset:
        new_data_l = Data(x=data_l.x,
                          edge_index=data_l.edge_index,
                          # y=1 if data_l.y > data_r.y else -1,
                          y=data_l.y - data_r.y,
                          domain=data_l.domain,
                          problem=data_l.problem,
                          pair_x=data_r.x,
                          pair_y=data_r.y,
                          pair_edge_index=data_r.edge_index,
                          )
        new_data_r = Data(x=data_r.x,
                          edge_index=data_r.edge_index,
                          y=1 if data_r.y > data_l.y else -1,
                          domain=data_r.domain,
                          problem=data_r.problem,
                          pair_x=data_l.x,
                          pair_y=data_l.y,
                          pair_edge_index=data_l.edge_index,
                          )

        # new_dataset.extend([new_data_l, new_data_r])
        new_dataset.extend([new_data_l])

    return new_dataset


def get_by_problem_dataloaders_from_args(args):
    model_name = args.model
    batch_size = args.batch_size
    domain = args.domain
    rep = args.rep
    max_nodes = args.max_nodes
    cutoff = args.cutoff
    small_train = args.small_train
    strategy = args.strategy
    num_workers = 0
    pin_memory = True

    dataset: List[List[Data]] = get_graph_data(domain=domain, representation=rep, paired=True)
    new_dataset = []
    index_list = []
    trainset = []
    valset = []
    valid_count = 0
    for i, datalist in enumerate(dataset):
        new_datalist = preprocess_data(model_name, data_list=datalist, c_hi=cutoff, n_hi=max_nodes,
                                       small_train=small_train)
        if len(new_datalist) < 2:
            continue
        new_datalist = add_features(new_datalist, args, idx=valid_count)
        new_dataset += new_datalist
        index_list += [valid_count] * len(datalist)
        get_stats(dataset=new_datalist, desc="Whole dataset")
        trains, vals = train_test_split(new_datalist, test_size=0.15, random_state=4550)
        trainset += trains
        valset += vals
        valid_count += 1
    train_per_class_sample_indices = ByProblemDataset(trainset,
                                                      valid_count).per_class_sample_indices()
    train_batch_sampler = BatchSampler(train_per_class_sample_indices,
                                       batch_size=batch_size)

    train_loader = DataLoader(
        trainset,
        # batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        # sampler=train_sampler,
        batch_sampler=train_batch_sampler
    )

    val_per_class_sample_indices = ByProblemDataset(valset,
                                                    valid_count).per_class_sample_indices()
    val_batch_sampler = BatchSampler(val_per_class_sample_indices,
                                     batch_size=batch_size)

    val_loader = DataLoader(
        valset,
        # batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        # sampler=train_sampler,
        batch_sampler=val_batch_sampler
    )
    get_stats(dataset=trainset, desc="Train set")
    get_stats(dataset=valset, desc="Val set")
    print("train size:", len(trainset))
    print("validation size:", len(valset))
    #
    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
    #                           num_workers=num_workers)
    # val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
    #                         num_workers=num_workers)

    return train_loader, val_loader


class ByProblemDataset(Dataset):
    def __init__(self, data: List[Data], num_classes):
        super().__init__()
        self.data = data
        self.data_len = len(data)

        indices = [[] for _ in range(num_classes)]

        for i, point in tqdm(enumerate(data), total=len(data), miniters=1,
                             desc='Building class indices dataset..'):
            indices[point.p_idx].append(i)

        self.indices = indices

    def per_class_sample_indices(self):
        return self.indices

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_len


class BatchSampler:
    def __init__(self, per_class_sample_indices, batch_size):
        # classes is a list of lists where each sublist refers to a class and contains
        # the sample ids that belond to this class
        self.per_class_sample_indices = per_class_sample_indices
        # self.n_batches = sum([len(x) for x in per_class_sample_indices]) // batch_size
        self.n_batches = len(per_class_sample_indices)
        self.min_class_size = min([len(x) for x in per_class_sample_indices])
        assert self.min_class_size > 0, "some problems with no samples"
        self.batch_size = batch_size
        self.class_range = list(range(len(self.per_class_sample_indices)))
        random.shuffle(self.class_range)

    def __iter__(self):
        for j in range(self.n_batches):
            if j < len(self.class_range):
                batch_class = self.class_range[j]
            else:
                batch_class = random.choice(self.class_range)
            # if self.batch_size <= len(self.per_class_sample_indices[batch_class]):
            #     batch = np.random.choice(self.per_class_sample_indices[batch_class], self.batch_size, replace=False)
            # else:
            batch = self.per_class_sample_indices[batch_class]
            yield batch

    def __len__(self):
        return self.n_batches * self.batch_size

    def n_batches(self):
        return self.n_batches
