
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
    parser.add_argument('--aggr', type=str, default="max")
    parser.add_argument('--pool', type=str, default="sum")
    parser.add_argument('--drop', type=float, default=0.0,
                        help="probability of an element to be zeroed")
    parser.add_argument('--vn', action='store_true',
                        help="use virtual nodes (doubles runtime)")

    # optimisation params
    parser.add_argument('--loss', type=str, choices=["mse", "wmse", "pemse"], default="mse")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-limit', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
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

def main():
    parser = create_parser()
    args = parser.parse_args()
    # configuration.check_config(args)
    print_arguments(args)
    # args.method = Method.from_str(args.method)
    if args.method != "goose":
        assert args.model != "RGNN", "Are you using ranker model?"

    print(f"testing {args.domain.split('-', 1)[1]}, model {args.save_file}")
    test.domain_test(args.domain.split("-", 1)[1], args.test_files, args.save_file, "test", log_root=args.log_root)
