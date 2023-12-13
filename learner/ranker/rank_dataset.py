import itertools
import os
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from representation import REPRESENTATIONS
from util.stats import get_stats

_DOWNWARD = "./../planners/downward/fast-downward.py"
_POWERLIFTED = "./../planners/powerlifted/powerlifted.py"
DATA_DIR = "./../data/ipc23/ranker"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
class ByProblemDataset(Dataset):
    def __init__(self, data: List, num_classes):
        super().__init__()
        self.data = data
        self.data_len = len(data)

        indices = [[] for _ in range(num_classes)]

        # for i, point in tqdm(enumerate(data), total=len(data), miniters=1,
        #                      desc='Building class indices dataset..'):
        # print('Building class indices dataset..')
        for i, point in enumerate(data):
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
        return self.n_batches

    def n_batches(self):
        return self.n_batches

    def n_samples(self):
        return sum([len(x) for x in self.per_class_sample_indices])


def get_plan_info(domain_pddl, problem_pddl, plan_file, args):
    planner = args.planner

    states = []
    actions = []

    with open(plan_file, "r") as f:
        for line in f.readlines():
            if ";" in line:
                continue
            actions.append(line.replace("\n", ""))

    aux_garbage = repr(hash((domain_pddl, problem_pddl, plan_file, repr(args))))
    aux_garbage = aux_garbage.replace("-", "n")
    state_output_file = aux_garbage + ".states"
    sas_file = aux_garbage + ".sas"

    cmd = {
        "pwl": f"export PLAN_PATH={plan_file} "
        + f"&& {_POWERLIFTED} -d {domain_pddl} -i {problem_pddl} -s perfect "
        + f"--plan-file {state_output_file}",
        "fd": f"export PLAN_INPUT_PATH={plan_file} "
        + f"&& export STATES_OUTPUT_PATH={state_output_file} "
        + f"&& {_DOWNWARD} --sas-file {sas_file} {domain_pddl} {problem_pddl} "
        + f"--search 'perfect_with_siblings([blind()])'",  # need filler h
    }[planner]
    output = os.popen(cmd).readlines()
    if output:
        pass  # this is so syntax highlighting sees `output`
    # os.system(cmd)
    with open(state_output_file, "r") as f:
        state_and_siblings = []
        for line in f.readlines():
            if line == "\n":
                states.append(state_and_siblings)
                state_and_siblings = []
                continue
            if ";" in line:
                continue
            line = line.replace("\n", "")
            s = set()
            for fact in line.split():
                if "(" not in fact:
                    lime = f"({fact})"
                else:
                    pred = fact[: fact.index("(")]
                    fact = fact.replace(pred + "(", "").replace(")", "")
                    args = fact.split(",")[:-1]
                    lime = "(" + " ".join([pred] + args) + ")"
                s.add(lime)
            state_and_siblings.append(s)

    if os.path.exists(sas_file):
        os.remove(sas_file)
    if os.path.exists(state_output_file):
        os.remove(state_output_file)

    schema_cnt = {}
    for action in actions:
        schema = action.replace("(", "").split()[0]
        if schema not in schema_cnt:
            schema_cnt[schema] = 0
        schema_cnt[schema] += 1

    ret = []
    for i, state in enumerate(states):
        if i == len(actions):
            ret.append((state, {0:0}))
        else:
            action = actions[i]
            schema = action.replace("(", "").split()[0]
            ret.append((state, schema_cnt.copy()))
            schema_cnt[schema] -= 1
    return ret

def get_tensor_graphs_from_plans_by_prob(args):
    print("Generating graphs from plans...")
    graphs = []

    representation = args.rep
    domain_pddl = args.domain_pddl
    tasks_dir = args.tasks_dir
    plans_dir = args.plans_dir

    for plan_file in sorted(list(os.listdir(plans_dir))):
        problem_pddl = f"{tasks_dir}/{plan_file.replace('.plan', '.pddl')}"
        assert os.path.exists(problem_pddl), problem_pddl
        plan_file = f"{plans_dir}/{plan_file}"
        rep = REPRESENTATIONS[representation](domain_pddl=domain_pddl, problem_pddl=problem_pddl)
        rep.convert_to_pyg()
        plan = get_plan_info(domain_pddl, problem_pddl, plan_file, args)

        graph_per_prob = []
        for i, (states, schema_cnt) in enumerate(plan):
            for j, state in enumerate(states):
                state = rep.str_to_state(state)
                x, edge_index = rep.state_to_tensor(state)
                y = sum(schema_cnt.values())
                graph = Data(x=x, edge_index=edge_index, y=y, coord_x=i, coord_y=j)
                graph_per_prob.append(graph)

        graphs.append(graph_per_prob)

    print("Graphs generated!")
    return graphs

def get_loaders_from_args_rank(args):
    batch_size = args.batch_size
    small_train = args.small_train
    data_dir = Path(f"{DATA_DIR}/{args.domain_pddl.split('/')[-2]}.data")
    if data_dir.is_file():
        print(f"Loading graphs from {data_dir}...")
        dataset = torch.load(data_dir)
    else:
        dataset = get_tensor_graphs_from_plans_by_prob(args)
    torch.save(dataset, data_dir)

    if small_train:
        random.seed(123)
        dataset = random.sample(dataset, k=10)

    val_interval = round(1 / args.val_ratio)
    train_valid_count = 0
    val_valid_count = 0
    trainset = []
    valset = []
    for i, problem in enumerate(dataset):
        if i % val_interval == 0:
            for data in problem:
                setattr(data, "p_idx", val_valid_count)
            valset.extend(problem)
            val_valid_count += 1
        else:
            for data in problem:
                setattr(data, "p_idx", train_valid_count)
            trainset.extend(problem)
            train_valid_count += 1

    train_batch_sampler = BatchSampler(ByProblemDataset(trainset,
                                                      train_valid_count).per_class_sample_indices(),
                                       batch_size=batch_size)
    train_loader = DataLoader(
        trainset,
        pin_memory=True,
        batch_sampler=train_batch_sampler
    )
    val_batch_sampler = BatchSampler(ByProblemDataset(valset,
                                                    val_valid_count).per_class_sample_indices(),
                                     batch_size=batch_size)

    val_loader = DataLoader(
        valset,
        pin_memory=True,
        batch_sampler=val_batch_sampler
    )
    get_stats(dataset=list(itertools.chain.from_iterable(dataset)), desc="Whole dataset")
    get_stats(dataset=trainset, desc="Train set")
    get_stats(dataset=valset, desc="Val set")
    print("train size:", len(trainset))
    print("validation size:", len(valset))


    return train_loader, val_loader