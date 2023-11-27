""" Main training pipeline script. """

import os
import pickle
import sys
import time
import argparse
import numpy as np
import representation
import kernels
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, log_loss, mean_squared_error
from kernels.wrapper import MODELS
from dataset.dataset_kernel import get_dataset_from_args, get_deadend_dataset_from_args
from util.save_load import print_arguments, save_kernel_model
from util.metrics import f1_macro
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS

import warnings

warnings.filterwarnings("ignore")

_SCORING_HEURISTIC = {
    "mse": mean_squared_error,
    "f1_macro": f1_macro,
}
_SCORING_DEADENDS = {
    "ll": log_loss,
    "f1_macro": f1_macro,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # ml model arguments
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        choices=[None] + MODELS,
        help="ML model. Use the default (None) when using the script to generate just the data and not training the models.",
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
        "--deadends",
        action="store_true",
        help="learn dead ends",
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
        type=str,
        choices=kernels.GRAPH_FEATURE_GENERATORS,
        help="wl algorithm to use",
    )
    parser.add_argument(
        "-l",
        "--iterations",
        type=int,
        default=5,
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
    parser.add_argument(
        "--data-save-file",
        type=str,
        default=None,
        help="save file for data; if this option is provided, training is skipped",
    )
    parser.add_argument(
        "--data-load-file",
        type=str,
        default=None,
        help="load file for data; if this option is provided, data generation is skipped",
    )

    args = parser.parse_args()

    if args.data_save_file is None and args.model is None:
        print("error: -m/--model is required when training")
        exit(-1)

    domain = args.domain
    args.domain_pddl = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    args.tasks_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy"
    args.plans_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training_plans"

    data_load_file = args.data_load_file
    if data_load_file is not None:
        # replaces parsed args by args loaded from file
        assert os.path.exists(data_load_file), data_load_file
        if args.model is None:
            print("error: -m/--model is required when loading data")
            sys.exit(-1)

        model = args.model
        a = args.a
        C = args.C
        e = args.e
        model_save_file = args.model_save_file

        with open(data_load_file, "rb") as inp:
            data = pickle.load(inp)
            args = data["train_args"]
        print(f"using all data args in {data_load_file}")

        args.data_load_file = data_load_file
        args.data_save_file = None

        args.model = model
        args.a = a
        args.C = C
        args.e = e
        args.model_save_file = model_save_file
    else:
        if args.domain is None or args.rep is None or args.features is None:
            parser.print_help()
            print(
                "error: the following arguments are required: -d/--domain, -r/--rep, -k/--features"
            )
            sys.exit(-1)

    return args


def main():
    args = parse_args()
    print_arguments(args)
    np.random.seed(args.seed)

    data_save_file = args.data_save_file
    data_load_file = args.data_load_file
    assert (
        data_save_file is None or data_load_file is None
    ), "cannot provide both save and load data files"

    predict_deadends = args.deadends

    # class decides whether to use classifier or regressor
    model = kernels.KernelModelWrapper(args)
    model.train()
    scoring = _SCORING_DEADENDS if predict_deadends else _SCORING_HEURISTIC

    if data_load_file is not None:
        print(f"loading X and y from {data_load_file}")
        with open(data_load_file, "rb") as inp:
            data = pickle.load(inp)
            X_train = data["X_train"]
            X_val = data["X_val"]
            y_train = data["y_train"]
            y_val = data["y_val"]
    else:
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

        # ## uncomment this and breakpoint at EOF to test correctness of cpp implementation
        # graphs_train = graphs
        # graphs_val = graphs
        # y_train = y_true
        # y_val = y_true

        print(f"Setting up training data...")
        t = time.time()
        train_histograms = model.compute_histograms(graphs_train)
        n_train_nodes = sum(len(G.nodes) for G in graphs_train)
        print(f"Initialised {args.features} for {len(graphs_train)} graphs")
        print(f"Collected {model.n_colours_} colours over {n_train_nodes} nodes")
        X_train = model.get_matrix_representation(graphs_train, train_histograms)
        print(f"Set up training data in {time.time()-t:.2f}s")

        print(f"Setting up validation data...")
        model.eval()
        t = time.time()
        val_histograms = model.compute_histograms(graphs_val)
        X_val = model.get_matrix_representation(graphs_val, val_histograms)
        print(f"Set up validation data in {time.time()-t:.2f}s")
        n_hit_colours = model.get_hit_colours()
        n_missed_colours = model.get_missed_colours()
        print(f"hit colours: {n_hit_colours}")
        print(f"missed colours: {n_missed_colours}")
        print(f"ratio hit/all colours: {n_hit_colours/(n_hit_colours+n_missed_colours):.2f}")

    # decide to save data or not
    # if save data, skip training
    if data_save_file is not None:
        save_dir = os.path.dirname(data_save_file)
        if len(save_dir) > 0:
            os.makedirs(save_dir, exist_ok=True)
        print(f"saving train and validation X and y to {data_save_file}")
        with open(data_save_file, "wb") as outp:
            data = {
                "train_args": args,
                "X_train": X_train,
                "X_val": X_val,
                "y_train": y_train,
                "y_val": y_val,
            }
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        return

    # training
    print(f"Training on entire {args.domain} for {args.model}...")
    t = time.time()
    model.fit(X_train, y_train)
    print(f"Model training completed in {time.time()-t:.2f}s")

    # predict on train and val sets
    print("Predicting...")
    t = time.time()
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
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

    # breakpoint()


if __name__ == "__main__":
    main()
