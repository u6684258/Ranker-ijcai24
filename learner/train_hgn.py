
import time

from typing import List

import argparse

import test_hgn
import representation


from tqdm.auto import tqdm, trange

from representation.hypergraph_nets.features.global_features import EmptyGlobalFeatureMapper
from representation.hypergraph_nets.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from representation.hypergraph_nets.features.node_features import PropositionInStateAndGoal
from util.metrics import SearchMetrics, SearchState
from util.stats import *
from util.save_load import *
from util import train, evaluate, Namespace
from dataset.dataset import get_loaders_from_args_gnn, get_new_dataloader_each_epoch, get_paired_dataloaders_from_args, \
    get_by_train_val_dataloaders_from_args, get_by_train_val_dataloaders_for_hgn
from util.train_eval import train_ranker, evaluate_ranker, train_hgn, evaluate_hgn


# TMP_FILE = "tmp/"
# Path(TMP_FILE).mkdir(parents=True, exist_ok=True)
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

    # optimisation params
    parser.add_argument('--loss', type=str, choices=["mse", "wmse", "pemse"], default="mse")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-limit', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--reduction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)

    # data arguments
    parser.add_argument('-r', '--rep', type=str, required=True, choices=representation.REPRESENTATIONS)
    parser.add_argument('-n', '--max-nodes', type=int, default=-1,
                        help="max nodes for generating graphs (-1 means no bound)")
    parser.add_argument('-c', '--cutoff', type=int, default=-1,
                        help="max cost to learn (-1 means no bound)")

    # save file
    parser.add_argument('--save-file', dest="save_file", type=str, default=None)
    parser.add_argument('--log-root', type=str, default="logs")
    # anti verbose
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.add_argument('--fast-train', action='store_true',
                        help="ignore some additional computation of stats, does not change the training algorithm")
    parser.add_argument('--test-only', action='store_true',
                        help="skip training")
    parser.add_argument('--test-files', type=str, default="test")
    parser.add_argument('--method',
                        choices=["hgn", "hgn_ranker"],
                        default="batched_ranker")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    # configuration.check_config(args)
    print_arguments(args)
    if args.method == "hgn":
        model_type = "hgn"
    elif args.method == "hgn_ranker":
        model_type = "ranker"
    else:
        model_type = "gnn"
    # cuda
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if not args.test_only:
        # init model
        if args.method == "hgn_ranker":
            train_loader, val_loader, hyps = get_by_train_val_dataloaders_for_hgn(args)
            args.out_feat = 64
            # args.in_feat = train_loader.dataset[0].x.shape[1]
            model_params = Namespace(
                model=args.model,
                receiver_k=hyps[0],
                sender_k=hyps[1],
                num_actns=hyps[2],
                num_props=hyps[3],
                hidden_size=32,
                latent_size=32,
                antisymmetric_activation=torch.nn.Sigmoid(),
                batch_size=args.batch_size,
                num_steps=args.nlayers,
                global_feature_mapper_cls=EmptyGlobalFeatureMapper,
                node_feature_mapper_cls=PropositionInStateAndGoal,
                hyperedge_feature_mapper_cls=ComplexHyperedgeFeatureMapper,
            )
        elif args.method == "hgn":
            train_loader, val_loader, hyps = get_by_train_val_dataloaders_for_hgn(args)
            args.out_feat = 1
            # args.in_feat = train_loader.dataset[0].x.shape[1]
            model_params = Namespace(
                model=args.model,
                device=device,
                receiver_k=hyps[0],
                sender_k=hyps[1],
                num_actns=hyps[2],
                num_props=hyps[3],
                hidden_size=32,
                latent_size=32,
                antisymmetric_activation=torch.nn.Sigmoid(),
                batch_size=args.batch_size,
                num_steps=args.nlayers,
                global_feature_mapper_cls=EmptyGlobalFeatureMapper,
                node_feature_mapper_cls=PropositionInStateAndGoal,
                hyperedge_feature_mapper_cls=ComplexHyperedgeFeatureMapper,
            )
        # train val pipeline
        print("Training...")
        best_val = None
        model_list = []
        for fold in range(5):
            model = GNNS[args.model](params=model_params).to(device)

            lr = args.lr
            reduction = args.reduction
            patience = args.patience
            epochs = args.epochs
            fast_train = args.fast_train

            # init optimiser
            criterion = LOSS["hgnloss"]()
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                                   mode='min',
                                                                   verbose=True,
                                                                   factor=reduction,
                                                                   patience=patience)
            print(f"model size (#params): {model.get_num_parameters()}")

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
                    train_stats = train_hgn(model, device, train_loader, criterion, optimiser, fast_train=fast_train)
                    train_loss = train_stats['loss']
                    val_stats = evaluate_hgn(model, device, val_loader, criterion, fast_train=fast_train)
                    val_loss = val_stats['loss']
                    scheduler.step(val_loss)

                    # take model weights corresponding to best combined metric # we just take val_loss
                    combined_metric = val_loss
                    if combined_metric < best_metric:
                        best_metric = combined_metric
                        best_dict = model
                        best_epoch = e

                    if fast_train:  # does not compute metrics like f1 score
                        desc = f"epoch {e}, " \
                               f"train_loss {train_loss:.2f}, " \
                               f"val_loss {val_loss:.2f}, " \
                               f"time {time.time() - t:.1f}"
                    else:
                        desc = f"epoch {e}, " \
                               f"train_loss {train_loss:.2f}, " \
                               f"val_loss {val_loss:.2f}, " \
                                f"train_f1 {train_stats['f1']:.1f}, " \
                                f"val_f1 {val_stats['f1']:.1f}, " \
                                f"train_acc {train_stats['acc']}, " \
                                f"val_acc {val_stats['acc']}, " \
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

            if best_dict is not None:
                if best_val is not None:
                    results: List[SearchMetrics] = test_hgn.domain_test(args.domain.split("-")[1], "val", args.save_file, log_root=args.log_root, timeout=300, model_type=model_type)
                    nodes_expanded = len([x.nodes_expanded for x in results]) / len(results)
                    if nodes_expanded < best_val:
                        best_val = nodes_expanded
                        print(f"best_avg_loss {best_metric:.8f} at fold {fold} epoch {best_epoch}")
                        args.best_metric = best_metric
                        save_hgn_model(best_dict, args)
                else:
                    args.best_metric = best_metric
                    save_hgn_model(best_dict, args)
                    results: List[SearchMetrics] = test_hgn.domain_test(args.domain.split("-")[1], "val", args.save_file, log_root=args.log_root, timeout=300, model_type=model_type)
                    best_val = len([x.nodes_expanded for x in results]) / len(results)

            else:
                save_hgn_model(model, args)

    print("testing...")
    test_hgn.domain_test(args.domain.split("-")[1], args.test_files, args.save_file, "test", log_root=args.log_root, model_type=model_type)

    return


if __name__ == "__main__":
    main()
