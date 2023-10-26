
import time

from typing import List

import argparse

import test
import representation


from tqdm.auto import tqdm, trange

from util.metrics import SearchMetrics, SearchState
from util.stats import *
from util.save_load import *
from util import train, evaluate
from dataset.dataset import get_loaders_from_args_gnn, get_new_dataloader_each_epoch, get_paired_dataloaders_from_args, \
    get_by_train_val_dataloaders_from_args
from util.train_eval import train_ranker, evaluate_ranker

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
    parser.add_argument('--small-train', action="store_true",
                        help="Small train set: useful for debugging.")

    # save file
    parser.add_argument('--save-file', dest="save_file", type=str, default=None)
    parser.add_argument('--log-root', type=str, default="log")
    # anti verbose
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.add_argument('--fast-train', action='store_true',
                        help="ignore some additional computation of stats, does not change the training algorithm")
    parser.add_argument('--test-only', action='store_true',
                        help="skip training")
    parser.add_argument('--test-files', type=str, default="test")
    parser.add_argument('--method',
                        choices=["goose", "ranker", "batched_ranker", "rnd_ranker", "batched_coord_ranker", "pretrained"],
                        default="batched_ranker")
    return parser


RANKER_GROUP = ["ranker", "batched_ranker", "rnd_ranker", "batched_coord_ranker", "pretrained"]


def get_distribution_of_pred_acc(pred, true, index):
    # diff = true - pred
    acc = torch.round(pred / true)
    accum = [0 for _ in range(int(max(index).item())+1)]
    count = [0 for _ in range(int(max(index).item())+1)]
    for i, ind in enumerate(index):
        accum[ind] += acc[i]
        count[ind] += 1

    print(np.array(accum) / np.array(count))
    print(np.array(count))


    plt.bar(np.arange(len(accum)), np.array(accum) / np.array(count),
            color='maroon',
            width=0.4)

    plt.xlabel("h*")
    plt.ylabel("average accuracy")
    plt.title("accuracy by h*")
    plt.show()

    plt.bar(np.arange(len(accum)), np.array(count),
            color='maroon',
            width=0.4)

    plt.xlabel("h*")
    plt.ylabel("count")
    plt.title("count by h*")
    plt.show()


def main():
    parser = create_parser()
    args = parser.parse_args()
    # configuration.check_config(args)
    print_arguments(args)
    # args.method = Method.from_str(args.method)
    if args.method != "goose":
        assert args.model != "RGNN", "Are you using ranker model?"
    # cuda
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if not args.test_only:
        # init model
        if args.method == "ranker":
            train_loader, val_loader = get_paired_dataloaders_from_args(args)
            print("Don't use this parameter!")
            args.out_feat = 64
            args.in_feat = train_loader.dataset[0].x.shape[1]
        elif args.method == "batched_ranker":
            train_loader, val_loader = get_by_train_val_dataloaders_from_args(args)
            args.out_feat = 64
            args.in_feat = train_loader.dataset[0].x.shape[1]
        elif args.method == "batched_coord_ranker" or args.method == "pretrained":
            train_loader, val_loader = get_by_train_val_dataloaders_from_args(args)
            args.out_feat = 64
            args.in_feat = train_loader.dataset[0].x.shape[1]
        elif args.method == "goose":
            train_loader, val_loader = get_by_train_val_dataloaders_from_args(args)
            args.out_feat = 1
            args.in_feat = train_loader.dataset[0].x.shape[1]
        elif args.method == "rnd_ranker":
            generator = get_new_dataloader_each_epoch(args)
            args.out_feat = 64
            args.in_feat = generator.dataset[0].x.shape[1]
        else:
            raise Exception("Invalid training method!")
        args.n_edge_labels = representation.REPRESENTATIONS[args.rep].n_edge_labels
        model_params = arg_to_params(args)

        # train val pipeline
        print("Training...")
        best_val = None
        model_list = []
        for fold in range(3):
            model = GNNS[args.model](params=model_params).to(device)
            if args.method == "pretrained":
                pretrained_model, _ = load_gnn_model(args.domain)
                pretrained_model.model.mlp = model.model.mlp
                model.model = pretrained_model.model
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
                    if args.method == "rnd_ranker":
                        train_loader, val_loader = generator.gen_new_dataloaders()
                    elif args.method in RANKER_GROUP:
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

                    # get_distribution_of_pred_acc(train_stats["pred"], train_stats["true"], train_stats["index"])
                    # get_distribution_of_pred_acc(val_stats["pred"], val_stats["true"], val_stats["index"])
            except KeyboardInterrupt:
                print("Early stopping due to keyboard interrupt!")
            # if best_dict is not None:
            #     model_list.append((best_metric, best_dict))
            # save model parameters
            if best_dict is not None:
                if best_val is not None:
                    results: List[SearchMetrics] = test.domain_test(args.domain.split("-")[1], "val", args.save_file, log_root=args.log_root, timeout=300)
                    nodes_expanded = len([x.nodes_expanded for x in results]) / len(results)
                    if nodes_expanded < best_val:
                        best_val = nodes_expanded
                        print(f"best_avg_loss {best_metric:.8f} at fold {fold} epoch {best_epoch}")
                        args.best_metric = best_metric
                        save_gnn_model_from_dict(best_dict, args)
                else:
                    args.best_metric = best_metric
                    save_gnn_model_from_dict(best_dict, args)
                    results: List[SearchMetrics] = test.domain_test(args.domain.split("-")[1], "val", args.save_file, log_root=args.log_root, timeout=300)
                    best_val = len([x.nodes_expanded for x in results]) / len(results)

            else:
                save_gnn_model(model, args)

    print("testing...")
    test.domain_test(args.domain.split("-")[1], args.test_files, args.save_file, "test", log_root=args.log_root)

    return


if __name__ == "__main__":
    main()
