import os
import time
from enum import Enum
from typing import List, Set

import torch
import argparse
import torch_geometric
import random
import numpy as np
import test
import representation

from models import *
from tqdm.auto import tqdm, trange
from gnns.loss import LOSS
from gnns import GNNS
from util.metrics import SearchMetrics, SearchState
from util.stats import *
from util.save_load import *
from util import train, evaluate
from dataset.dataset import get_loaders_from_args_gnn, \
    get_by_problem_dataloaders_from_args, get_new_dataloader_each_epoch
from util.train_eval import train_ranker, evaluate_ranker


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('-d', '--domain', default="goose-di")
    parser.add_argument('-t', '--task', default='h', choices=["h", "a"],
                        help="predict value or action (currently only h is supported)")

    # model params
    parser.add_argument('-m', '--model', type=str, required=True, choices=GNNS)
    parser.add_argument('-L', '--nlayers', type=int, default=7)
    parser.add_argument('-H', '--nhid', type=int, default=64)
    parser.add_argument('--share-layers', action='store_true')
    parser.add_argument('--aggr', type=str, default="mean")
    parser.add_argument('--pool', type=str, default="sum")
    parser.add_argument('--drop', type=float, default=0.0,
                        help="probability of an element to be zeroed")
    parser.add_argument('--vn', action='store_true',
                        help="use virtual nodes (doubles runtime)")

    # optimisation params
    parser.add_argument('--loss', type=str, choices=["mse", "wmse", "pemse"], default="mse")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-limit', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--reduction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)

    # data arguments
    parser.add_argument('-r', '--rep', type=str, required=True, choices=representation.REPRESENTATIONS)
    parser.add_argument('-n', '--max-nodes', type=int, default=-1,
                        help="max nodes for generating graphs (-1 means no bound)")
    parser.add_argument('-c', '--cutoff', type=int, default=-1,
                        help="max cost to learn (-1 means no bound)")
    parser.add_argument('--small-train', action="store_true",
                        help="Small train set: useful for debugging.")

    # save file
    parser.add_argument('--save-file', dest="save_file", type=str, default=None)

    # anti verbose
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.add_argument('--fast-train', action='store_true',
                        help="ignore some additional computation of stats, does not change the training algorithm")
    parser.add_argument('--test-files', type=str, default="test")
    parser.add_argument('--method', choices=Method,
                        type=lambda val: Method.from_str(val),
                        default=Method.BAT_RANKER.name)
    return parser


class _BasePlanningEnum(Enum):
    @classmethod
    def member_names(cls) -> Set[str]:
        return {member.name for member in cls}

    @classmethod
    def from_str(cls, the_str: str):
        return cls._from_str(the_str)

    @classmethod
    def _from_str(cls, the_str: str, error_str="Unknown member {}"):
        for member in cls:
            if member.name == the_str:
                return member

        raise ValueError(error_str.format(the_str))

    def __str__(self):
        return self.value


class Method(_BasePlanningEnum):
    GOOSE = "goose"
    RANKER = "ranker"
    BAT_RANKER = "batched_ranker"
    RND_RANKER = "ranker_random"


RANKER_GROUP = [Method.RANKER, Method.BAT_RANKER, Method.RND_RANKER]


def main():
    parser = create_parser()
    args = parser.parse_args()
    # configuration.check_config(args)
    print_arguments(args)
    if args.method != Method.GOOSE:
        assert args.model == "RGNNRANK" or args.model == "RGNNBATRANK", "Are you using ranker model?"
    # cuda
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # init model
    if args.method == Method.RANKER:
        train_loader, val_loader = get_by_problem_dataloaders_from_args(args)
        print("Don't use this parameter!")
        args.out_feat = 64
        args.in_feat = train_loader.dataset[0].x.shape[1]
    elif args.method == Method.BAT_RANKER:
        train_loader, val_loader = get_by_problem_dataloaders_from_args(args)
        args.out_feat = 64
        args.in_feat = train_loader.dataset[0].x.shape[1]
    elif args.method == Method.GOOSE:
        train_loader, val_loader = get_loaders_from_args_gnn(args)
        args.out_feat = 1
        args.in_feat = train_loader.dataset[0].x.shape[1]
    elif args.method == Method.RND_RANKER:
        generator = get_new_dataloader_each_epoch(args)
        args.out_feat = 64
        args.in_feat = generator.dataset[0].x.shape[1]
    else:
        raise Exception("Invalid training method!")
    args.n_edge_labels = representation.REPRESENTATIONS[args.rep].n_edge_labels
    model_params = arg_to_params(args)
    model = GNNS[args.model](params=model_params).to(device)

    lr = args.lr
    reduction = args.reduction
    patience = args.patience
    epochs = args.epochs
    loss_fn = args.loss
    fast_train = args.fast_train

    # init optimiser
    criterion = LOSS[loss_fn]()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                           mode='min',
                                                           verbose=True,
                                                           factor=reduction,
                                                           patience=patience)

    print(f"model size (#params): {model.get_num_parameters()}")

    # train val pipeline
    print("Training...")
    best_val = None
    for fold in range(5):
        best_dict = None
        best_saved_model = None
        best_metric = float('inf')
        try:
            if args.tqdm:
                pbar = trange(epochs)
            else:
                pbar = range(epochs)
            best_epoch = 0
            for e in pbar:
                if args.method == Method.RND_RANKER:
                    train_loader, val_loader = generator.gen_new_dataloaders()
                t = time.time()
                if args.method in RANKER_GROUP:
                    train_stats = train_ranker(model, device, train_loader, criterion, optimiser, fast_train=fast_train)
                else:
                    train_stats = train(model, device, train_loader, criterion, optimiser, fast_train=fast_train)
                train_loss = train_stats['loss']
                if args.method in RANKER_GROUP:
                    val_stats = evaluate_ranker(model, device, val_loader, criterion, fast_train=fast_train)
                else:
                    val_stats = evaluate(model, device, val_loader, criterion, fast_train=fast_train)
                val_loss = val_stats['loss']
                scheduler.step(val_loss)

                # take model weights corresponding to best combined metric # we just take val_loss
                combined_metric = val_loss
                if combined_metric < best_metric:
                    best_metric = combined_metric
                    best_dict = model.model.state_dict()
                    best_epoch = e

                if fast_train:  # does not compute metrics like f1 score
                    desc = f"epoch {e}, " \
                           f"train_loss {train_loss:.2f}, " \
                           f"val_loss {val_loss:.2f}, " \
                           f"time {time.time() - t:.1f}"
                else:  # computes all metrics
                    # f"train_f1 {train_stats['f1']:.1f}, " \
                    # f"val_f1 {val_stats['f1']:.1f}, " \
                    # f"train_int {train_stats['interval']}, " \
                    # f"val_int {val_stats['interval']}, " \
                    # f"train_adm {train_stats['admis']:.1f}, " \
                    # f"val_adm {val_stats['admis']:.1f}, " \
                    desc = f"epoch {e}, " \
                           f"train_loss {train_loss:.2f}, " \
                           f"val_loss {val_loss:.2f}, " \
                           f"time {time.time() - t:.1f}"

                lr = optimiser.param_groups[0]['lr']
                if args.tqdm:
                    tqdm.write(desc)
                    pbar.set_description(desc)
                else:
                    print(desc)

                if lr < args.lr_limit:
                    print(f"Early stopping due to small lr: {lr}")
                    break
        except KeyboardInterrupt:
            print("Early stopping due to keyboard interrupt!")

        # save model parameters
        if best_dict is not None:
            if best_val is not None:
                results: List[SearchMetrics] = test.domain_test(args.domain.split("-")[1], "val", args.save_file)
                succ_rate = len([x.plan_length for x in results if x.search_state == SearchState.success]) / len(
                    results)
                if succ_rate > best_val:
                    best_val = succ_rate
                    print(f"best_avg_loss {best_metric:.8f} at fold {fold} epoch {best_epoch}")
                    args.best_metric = best_metric
                    save_gnn_model_from_dict(best_dict, args)
            else:
                args.best_metric = best_metric
                save_gnn_model_from_dict(best_dict, args)
                results: List[SearchMetrics] = test.domain_test(args.domain.split("-")[1], "val", args.save_file)
                best_val = len([x.plan_length for x in results if x.search_state == SearchState.success]) / len(
                    results)

        else:
            save_gnn_model(model, args)

    test.domain_test(args.domain.split("-")[1], args.test_files, args.save_file, "test")

    return


if __name__ == "__main__":
    main()
