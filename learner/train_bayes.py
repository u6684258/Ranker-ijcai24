""" Main training pipeline script. """

import os
import pickle
import sys
import time
import argparse
from matplotlib import pyplot as plt
import numpy as np
import representation
import kernels
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, log_loss, mean_squared_error
from kernels.wrapper import BAYESIAN_MODELS, MODELS
from dataset.dataset_kernel import get_dataset_from_args, get_deadend_dataset_from_args
from util.save_load import print_arguments, save_kernel_model
from util.metrics import f1_macro
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # ml model arguments
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="blr",
        choices=BAYESIAN_MODELS,
        help="Bayesian ML model.",
    )
    parser.add_argument(
        "--model-save-file", type=str, default=None, help="save file of model weights"
    )

    # data arguments
    parser.add_argument(
        "-d",
        "--domain",
        help="domain to learn domain knowledge for",
        choices=IPC2023_LEARNING_DOMAINS,
    )
    parser.add_argument(
        "-r",
        "--rep",
        type=str,
        default="ilg",
        choices=representation.REPRESENTATIONS,
        help="graph representation to use",
    )
    parser.add_argument(
        "-k",
        "--features",
        default="1wl",
        type=str,
        choices=kernels.GRAPH_FEATURE_GENERATORS,
        help="wl algorithm to use",
    )
    parser.add_argument(
        "-l",
        "--iterations",
        type=int,
        default=4,
        help="number of iterations for kernel algorithms",
    )
    parser.add_argument(
        "-p",
        "--prune",
        type=int,
        default=0,
        help="reduce feature sizes by discarding colours with total train count <= prune",
    )
    parser.add_argument(
        "--small-train",
        action="store_true",
        help="use small train set, useful for debugging",
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument("--planner", default="fd", choices=["fd", "pwl"])
    args = parser.parse_args()

    domain = args.domain

    return args


def main():
    args = parse_args()
    print_arguments(args)
    np.random.seed(args.seed)

    args.deadends = False  # TODO
    predict_deadends = args.deadends

    # class decides whether to use classifier or regressor
    model = kernels.KernelModelWrapper(args)
    model.train()
    scoring = {
        "mse": mean_squared_error,
        "f1_macro": f1_macro,
    }

    if predict_deadends:
        graphs, y_true = get_deadend_dataset_from_args(args)
        if len(graphs) == 0:
            print(f"No deadends to learn for {args.domain}!")
            print(f"Saving a redundant file...")
            args.model = "empty"
            model = kernels.KernelModelWrapper(args)
            save_kernel_model(model, args)
            return
    else:
        graphs, y_true = get_dataset_from_args(args)

    graphs_train, graphs_val, y_train, y_val = train_test_split(
        graphs, y_true, test_size=0.33, random_state=2023
    )

    print(f"Setting up training data...")
    t = time.time()
    train_histograms, train_ratio_seen_counts = model.compute_histograms(
        graphs_train, return_ratio_seen_counts=True
    )
    n_train_nodes = sum(len(G.nodes) for G in graphs_train)
    print(f"Initialised {args.features} for {len(graphs_train)} graphs")
    print(f"Collected {model.n_colours_} colours over {n_train_nodes} nodes")
    X_train = model.get_matrix_representation(graphs_train, train_histograms)
    print(f"Set up training data in {time.time()-t:.2f}s")

    print(f"Setting up validation data...")
    model.eval()
    t = time.time()
    val_histograms, val_ratio_seen_counts = model.compute_histograms(
        graphs_val, return_ratio_seen_counts=True
    )
    X_val = model.get_matrix_representation(graphs_val, val_histograms)
    print(f"Set up validation data in {time.time()-t:.2f}s")
    n_hit_colours = model.get_hit_colours()
    n_missed_colours = model.get_missed_colours()
    print(f"hit colours: {n_hit_colours}")
    print(f"missed colours: {n_missed_colours}")
    print(f"ratio hit/all colours: {n_hit_colours/(n_hit_colours+n_missed_colours):.2f}")

    # training
    print(f"Training on entire {args.domain} for {args.model}...")
    t = time.time()
    model.fit(X_train, y_train)
    print(f"Model training completed in {time.time()-t:.2f}s")

    # predict on train and val sets
    print("Predicting...")
    t = time.time()
    y_train_pred, y_train_pred_std = model.predict_with_std(X_train)
    y_val_pred, y_val_pred_std = model.predict_with_std(X_val)
    print(f"Predicting completed in {time.time()-t:.2f}s")

    # metrics
    print("Scores:")
    t = time.time()
    for metric in scoring:
        print(f"train_{metric}: {scoring[metric](y_train, y_train_pred):.2f}")
        print(f"val_{metric}: {scoring[metric](y_val, y_val_pred):.2f}")

    if predict_deadends:
        print("train confusion matrix:")
        print(confusion_matrix(y_train, y_train_pred))
        print("val confusion matrix:")
        print(confusion_matrix(y_val, y_val_pred))

    # save model
    save_kernel_model(model, args)

    # print % weights are zero
    try:
        n_zero_weights = model.get_num_zero_weights()
        n_weights = model.get_num_weights()
        print(f"zero_weights: {n_zero_weights}/{n_weights} = {n_zero_weights/n_weights:.2f}")
    except Exception as e:  # not possible for true kernel methods
        pass

    plt.scatter(val_ratio_seen_counts, y_val_pred_std)
    plt.xlim((0, 1))
    plt.ylim(bottom=0)
    save_dir = "plots_unseen"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{args.domain}_{args.model}.png", dpi=480)


if __name__ == "__main__":
    main()
