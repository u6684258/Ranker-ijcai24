import os
import time
from typing import List

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
    get_by_problem_dataloaders_from_args
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
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--reduction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2000)

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
    parser.add_argument('--batched-ranker', action='store_true')
    parser.add_argument('--ranker', action='store_true')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    # configuration.check_config(args)
    print_arguments(args)
    if args.ranker or args.batched_ranker:
        assert args.model == "RGNNRANK" or args.model == "RGNNBATRANK", "Are you using ranker model?"
        assert args.ranker != args.batched_ranker, "Cannot set ranker and batched_ranker at the same time"
    # cuda
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # init model
    if args.ranker:
        train_loader, val_loader = get_by_problem_dataloaders_from_args(args)
        print("Don't use this parameter!")
        args.out_feat = 64
    elif args.batched_ranker:
        train_loader, val_loader = get_by_problem_dataloaders_from_args(args)
        args.out_feat = 64
    else:
        train_loader, val_loader = get_loaders_from_args_gnn(args)
        args.out_feat = 1
    args.n_edge_labels = representation.REPRESENTATIONS[args.rep].n_edge_labels
    args.in_feat = train_loader.dataset[0].x.shape[1]
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
                t = time.time()
                if args.batched_ranker:
                    train_stats = train_ranker(model, device, train_loader, criterion, optimiser, fast_train=fast_train)
                else:
                    train_stats = train(model, device, train_loader, criterion, optimiser, fast_train=fast_train)
                train_loss = train_stats['loss']
                if args.batched_ranker:
                    val_stats = evaluate_ranker(model, device, val_loader, criterion, fast_train=fast_train)
                else:
                    val_stats = evaluate(model, device, val_loader, criterion, fast_train=fast_train)
                val_loss = val_stats['loss']
                scheduler.step(val_loss)

                # take model weights corresponding to best combined metric
                combined_metric = (train_loss + 2 * val_loss) / 3
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

                if lr < 1e-5:
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

    test.domain_test(args.domain.split("-")[1], args.test_files, args.save_file)

    return


if __name__ == "__main__":
    main()
