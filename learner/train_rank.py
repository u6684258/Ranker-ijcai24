""" Main training pipeline script. """

import time
import torch
import argparse
import representation
from gnns.loss import BCELoss, MSELoss
from gnns.gnn import Model
from gnns.train_eval import train, evaluate
from hgn.dataset_hgn import get_loaders_from_args_hgn
from hgn.hypergraph_nets.features.global_features import EmptyGlobalFeatureMapper
from hgn.hypergraph_nets.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from hgn.hypergraph_nets.features.node_features import PropositionInStateAndGoal
from hgn.model_hgn import HgnModel, HGNLoss
from hgn.rank_dataset_hgn import get_loaders_from_args_hgn_rank
from hgn.rank_model_hgn import HgnRankModel
from hgn.rank_train_eval_hgn import hgn_rank_evaluate, hgn_rank_train
from hgn.train_eval_hgn import hgn_train, hgn_evaluate
from ranker.rank_dataset import get_loaders_from_args_rank
from ranker.rank_model import RankModel
from ranker.rank_train_eval import rank_train, rank_evaluate
from util.stats import *
from util.save_load import *
from dataset.dataset_gnn import get_loaders_from_args_gnn
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=IPC2023_LEARNING_DOMAINS)

    # model params
    parser.add_argument("-m", "--model",
                        choices=["gnn", "gnn-rank", "hgn", "hgn-rank"],
                        default="gnn"
                        )
    parser.add_argument("-L", "--nlayers", type=int, default=4)
    parser.add_argument("-H", "--nhid", type=int, default=64)
    parser.add_argument("-E", "--nEmb", type=int, default=64, help="embedding dimension")
    parser.add_argument(
        "--aggr",
        type=str,
        default="mean",
        choices=["sum", "mean", "max"],
        help="mpnn aggregation function",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="pooling function for readout",
    )

    # optimisation params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--reduction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)

    # data arguments
    parser.add_argument(
        "-r",
        "--rep",
        type=str,
        required=True,
        choices=["slg", "flg", "llg", "ilg"],
        help="graph representation of planning tasks",
    )
    parser.add_argument(
        "-p",
        "--planner",
        type=str,
        default="fd",
        choices=["fd", "pwl"],
        help="for converting plans to states",
    )
    parser.add_argument(
        "--small-train",
        action="store_true",
        help="Small training set: useful for debugging.",
    )

    parser.add_argument(
        "-i",
        "--val-ratio",
        type=float,
        default=0.1,
        help="train-validation split ratio",
    )

    # save file
    parser.add_argument("--save-file", dest="save_file", type=str, default=None)

    # gpu device (if exists)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    domain = args.domain
    args.domain_pddl = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    args.tasks_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy"
    args.plans_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training_plans"

    return args


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    # cuda
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # init model
    if args.model == "gnn":
        train_loader, val_loader = get_loaders_from_args_gnn(args)
    elif args.model == "gnn-rank":
        train_loader, val_loader = get_loaders_from_args_rank(args)
    elif args.model == "hgn":
        train_loader, val_loader, m_re, m_se = get_loaders_from_args_hgn(args)
    elif args.model == "hgn-rank":
        train_loader, val_loader, m_re, m_se = get_loaders_from_args_hgn_rank(args)

    else:
        assert False, f"Invalid model type: {args.model}"

    if args.model == "gnn":
        args.n_edge_labels = len(train_loader.dataset[0].edge_index)
        args.in_feat = train_loader.dataset[0].x.shape[1]
        model_params = arg_to_params(args)
        model = Model(params=model_params).to(device)
    elif args.model == "gnn-rank":
        args.n_edge_labels = len(train_loader.dataset[0].edge_index)
        args.in_feat = train_loader.dataset[0].x.shape[1]
        model_params = arg_to_params(args)
        model_params["out_feat"] = args.nEmb
        model = RankModel(params=model_params).to(device)
    elif args.model == "hgn":
        model_params = {
            "model":args.model,
            "device":device,
            "receiver_k":m_re,
            "sender_k":m_se,
            "hidden_size":args.nEmb,
            "latent_size":args.nEmb,
            "antisymmetric_activation":torch.nn.Sigmoid(),
            "batch_size":args.batch_size,
            "num_steps":args.nlayers,
            "global_feature_mapper_cls":EmptyGlobalFeatureMapper,
            "node_feature_mapper_cls":PropositionInStateAndGoal,
            "hyperedge_feature_mapper_cls":ComplexHyperedgeFeatureMapper,
        }
        model = HgnModel(params=model_params).to(device)
    elif args.model == "hgn-rank":
        model_params = {
            "model": args.model,
            "device": device,
            "receiver_k": m_re,
            "sender_k": m_se,
            "hidden_size": args.nEmb,
            "latent_size": args.nEmb,
            "antisymmetric_activation": torch.nn.Sigmoid(),
            "batch_size": args.batch_size,
            "num_steps": args.nlayers,
            "global_feature_mapper_cls": EmptyGlobalFeatureMapper,
            "node_feature_mapper_cls": PropositionInStateAndGoal,
            "hyperedge_feature_mapper_cls": ComplexHyperedgeFeatureMapper,
        }
        model = HgnRankModel(params=model_params).to(device)
    else:
        assert False, f"Invalid model type: {args.model}"

    print(f"model size (#params): {model.get_num_parameters()}")

    # argument variables
    lr = args.lr
    reduction = args.reduction
    patience = args.patience
    epochs = args.epochs

    # init optimiser
    if args.model == "gnn":
        criterion = MSELoss()
    if args.model == "gnn-rank":
        criterion = MSELoss()
    elif args.model == "hgn":
        criterion = HGNLoss()
    elif args.model == "hgn-rank":
        criterion = HGNLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", verbose=True, factor=reduction, patience=patience
    )

    # train val pipeline
    print("device:", device)
    print("Training...")
    try:
        best_dict = None
        best_metric = float("inf")
        best_epoch = 0
        for e in range(epochs):
            t = time.time()
            if args.model == "gnn":
                train_stats = train(
                    model, device, train_loader, criterion, optimiser
                )
            elif args.model == "gnn-rank":
                train_stats = rank_train(
                    model, device, train_loader, criterion, optimiser
                )

            elif args.model == "hgn":
                train_stats = hgn_train(
                    model, device, train_loader, criterion, optimiser
                )
            elif args.model == "hgn-rank":
                train_stats = hgn_rank_train(
                    model, device, train_loader, criterion, optimiser
                )
            else:
                assert False, f"Invalid model type: {args.model}"
            train_loss = train_stats["loss"]
            if args.model == "gnn":
                val_stats = evaluate(model, device, val_loader, criterion)
            elif args.model == "gnn-rank":
                val_stats = rank_evaluate(model, device, val_loader, criterion)
            elif args.model == "hgn":
                val_stats = hgn_evaluate(
                    model, device, train_loader, criterion, optimiser
                )
            elif args.model == "hgn-rank":
                val_stats = hgn_rank_evaluate(
                    model, device, train_loader, criterion, optimiser
                )
            else:
                assert False, f"Invalid model type: {args.model}"
            val_loss = val_stats["loss"]
            scheduler.step(val_loss)

            # take model weights corresponding to best combined metric
            combined_metric = (train_loss + 2 * val_loss) / 3
            if combined_metric < best_metric:
                best_metric = combined_metric
                if args.model == "hgn" or args.model == "hgn-rank":
                    best_dict = model
                else:
                    best_dict = model.state_dict()
                best_epoch = e

            desc = f"epoch {e}, " \
                    f"time {time.time() - t:.1f}, " \
                    f"train_loss {train_loss:.2f}, " \
                    f"val_loss {val_loss:.2f} "
            print(desc)

            lr = optimiser.param_groups[0]["lr"]
            if lr < 1e-5:
                print(f"Early stopping due to small lr: {lr}")
                break
    except KeyboardInterrupt:
        print("Early stopping due to keyboard interrupt!")

    # save model parameters
    if best_dict is not None:
        print(f"best_avg_loss {best_metric:.8f} at epoch {best_epoch}")
        args.best_metric = best_metric
        if args.model == "hgn" or args.model == "hgn-rank":
            save_hgn_model(best_dict, args)
        else:
            save_gnn_model_from_dict(best_dict, args)
