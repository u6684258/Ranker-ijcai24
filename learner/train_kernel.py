""" Main training pipeline script. """

import os
import time
import argparse
import numpy as np
import representation
import kernels
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, log_loss, make_scorer, mean_squared_error
from kernels.wrapper import MODELS
from dataset.graphs_kernel import get_dataset_from_args, get_deadend_dataset_from_args
from util.save_load import print_arguments, save_kernel_model
from util.metrics import f1_macro
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS

import warnings

warnings.filterwarnings("ignore")

_CV_FOLDS = 5
_PLOT_DIR = "plots"
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
    # parser.add_argument("domain_pddl", help="path to domain pddl")
    # parser.add_argument("tasks_dir", help="path to training task directory")
    # parser.add_argument("plans_dir", help="path to training plan directory")

    parser.add_argument(
        "domain",
        help="domain to learn domain knowledge for",
        choices=IPC2023_LEARNING_DOMAINS,
    )

    parser.add_argument(
        "-d",
        "--deadends",
        action="store_true",
        help="learn dead ends",
    )

    parser.add_argument(
        "-r",
        "--rep",
        type=str,
        required=True,
        choices=representation.REPRESENTATIONS,
        help="graph representation to use",
    )

    parser.add_argument(
        "-k",
        "--features",
        type=str,
        required=True,
        choices=kernels.GRAPH_FEATURE_GENERATORS,
        help="graph representation to use",
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
        "-m", "--model", type=str, default="linear-svr", choices=MODELS, help="ML model"
    )
    parser.add_argument(
        "-a",
        type=float,
        default=1,
        help="L1 and L2 regularisation parameter of linear regression; strength is proportional to a",
    )
    parser.add_argument(
        "-C",
        type=float,
        default=1,
        help="regularisation parameter of SVR; strength is inversely proportional to C",
    )
    parser.add_argument(
        "-e",
        type=float,
        default=0.1,
        help="epsilon parameter in epsilon insensitive loss function of SVR",
    )

    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument("--planner", default="fd", choices=["fd", "pwl"])

    parser.add_argument("--save-file", type=str, default=None, help="save file of model weights")
    parser.add_argument(
        "--small-train",
        action="store_true",
        help="use small train set, useful for debugging",
    )

    parser.add_argument(
        "--matrix-save-file",
        type=str,
        default=None,
        help="save file for data; if this option is provided, training is skipped",
    )

    parser.add_argument(
        "--matrix-load-file",
        type=str,
        default=None,
        help="load file for data; if this option is provided, data generation is skipped",
    )

    args = parser.parse_args()
    domain = args.domain
    args.domain_pddl = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    args.tasks_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy"
    args.plans_dir = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training_plans"
    return args


def main():
    args = parse_args()
    print_arguments(args)
    np.random.seed(args.seed)

    matrix_save_file = args.matrix_save_file
    matrix_load_file = args.matrix_load_file

    predict_deadends = args.deadends

    if matrix_load_file is not None:
        # TODO, add some code that checks matrix is compatible with args
        print(f"loading X and y from {matrix_load_file}")
        data = np.load(matrix_load_file)
        X = data["X"]
        y_true = data["y"]
    else:
        if predict_deadends:
            graphs, y_true = get_deadend_dataset_from_args(args)
            scoring = _SCORING_DEADENDS
            if len(graphs) == 0:
                print(f"No deadends to learn for {args.domain}!")
                print(f"Saving a redundant file...")
                args.model = "empty"
                model = kernels.KernelModelWrapper(args)
                save_kernel_model(model, args)
                return
        else:
            graphs, y_true = get_dataset_from_args(args)
            scoring = _SCORING_HEURISTIC

        print(f"Setting up training data and initialising model...")
        t = time.time()
        # class decides whether to use classifier or regressor
        model = kernels.KernelModelWrapper(args)
        model.train()
        t = time.time()
        train_histograms = model.compute_histograms(graphs)
        print(f"Initialised {args.features} for {len(graphs)} graphs in {time.time() - t:.2f}s")
        print(
            f"Collected {model.n_colours_} colours over {sum(len(G.nodes) for G in graphs)} nodes"
        )
        X = model.get_matrix_representation(graphs, train_histograms)
        print(f"Set up training data in {time.time()-t:.2f}s")

    # decide to save data or not
    # if save data, skip training
    if matrix_save_file is not None:
        print(f"saving X and y to {matrix_save_file}")
        np.savez(matrix_save_file, X=X, y=y_true)
        return

    # training
    print(f"Training on entire {args.domain} for {args.model}...")
    t = time.time()
    model.fit(X, y_true)
    print(f"Model training completed in {time.time()-t:.2f}s")

    # metrics
    print("Predicting...")
    t = time.time()
    y_pred = model.predict(X)
    print(f"Predicting completed in {time.time()-t:.2f}s")
    print("Scoring...")
    t = time.time()
    for metric in scoring:
        score = scoring[metric](y_true, y_pred)
        print(f"train_{metric}: {score:.2f}")
    print(f"Scoring completed in {time.time()-t:.2f}s")

    if predict_deadends:
        print("confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    # save model
    save_kernel_model(model, args)

    # print % weights are zero
    try:
        n_zero_weights = model.get_num_zero_weights()
        n_weights = model.get_num_weights()
        print(f"zero_weights: {n_zero_weights}/{n_weights} = {n_zero_weights/n_weights:.2f}")
    except Exception as e:  # not possible for true kernel methods
        pass


if __name__ == "__main__":
    main()
