import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import kernels
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.svm import LinearSVR, SVR, LinearSVC, SVC
from sklearn.gaussian_process.kernels import DotProduct
from typing import Iterable, List, Optional, Dict, Tuple, Union
from representation import CGraph, Representation, REPRESENTATIONS
from planning import State
from kernels.base_kernel import Histogram, NO_EDGE, WlAlgorithm


MODELS = [
    # "linear-regression",
    "linear-svr",
    "ridge",
    "lasso",
    "rbf-svr",
    "quadratic-svr",
    "cubic-svr",
    "mlp",
]

BAYESIAN_MODELS = [
    "blr",  # bayesian linear regression
    "gp",  # gaussian process with dot product kernel
]

LINEAR_MODELS = [
    "gp", "linear-svr"
]

_MAX_MODEL_ITER = 1000000
_C = 1.0
_ALPHA = 1.0
_EPSILON = 0.1


class KernelModelWrapper:
    def __init__(self, args) -> None:
        super().__init__()
        if args.model == "empty":
            return  # when there are no dead ends to learn
        self.model_name = args.model
        self.wl_name = args.features

        self._kernel : WlAlgorithm = kernels.GRAPH_FEATURE_GENERATORS[args.features](iterations=args.iterations, prune=args.prune)

        self._iterations = args.iterations
        self._prune = args.prune

        self._rep_type = args.rep
        self._representation = None

        kwargs = {
            "max_iter": _MAX_MODEL_ITER,
        }

        self._deadends = args.deadends
        if self._deadends:  # deadends is binary classification
            self._model = {
                None: None,
                "linear-regression": LogisticRegression(penalty=None),
                "linear-svr": LinearSVC(dual="auto", C=_C, **kwargs),
                "lasso": LogisticRegression(  # l1 only works with liblinear and saga
                    penalty="l1", C=1 / _ALPHA, solver="liblinear", **kwargs
                ),
                "ridge": LogisticRegression(penalty="l2", C=1 / _ALPHA, **kwargs),
                "rbf-svr": SVC(kernel="rbf", C=_C, **kwargs),
                "quadratic-svr": SVC(kernel="poly", degree=2, C=_C, **kwargs),
                "cubic-svr": SVC(kernel="poly", degree=3, C=_C, **kwargs),
                "mlp": MLPClassifier(
                    hidden_layer_sizes=(64,),
                    batch_size=16,
                    learning_rate="adaptive",
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            }[self.model_name]
        else:  # heuristic is regression
            self._model = {
                None: None,
                "linear-regression": LinearRegression(),
                "linear-svr": LinearSVR(dual="auto", epsilon=_EPSILON, C=_C, **kwargs),
                "lasso": Lasso(alpha=_ALPHA, **kwargs),
                "ridge": Ridge(alpha=_ALPHA, **kwargs),
                "rbf-svr": SVR(kernel="rbf", epsilon=_EPSILON, C=_C, **kwargs),
                "quadratic-svr": SVR(kernel="poly", degree=2, epsilon=_EPSILON, C=_C, **kwargs),
                "cubic-svr": SVR(kernel="poly", degree=3, epsilon=_EPSILON, C=_C, **kwargs),
                "mlp": MLPRegressor(
                    hidden_layer_sizes=(64,),
                    batch_size=16,
                    learning_rate="adaptive",
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
                "blr": BayesianRidge(),
                "gp": GaussianProcessRegressor(kernel=DotProduct(), alpha=1e-8 if args.domain == "sokoban" else 1e-10),  
            }[self.model_name]

        self._train = True
        self._indices = None

    def train(self) -> None:
        """set train mode, not actually training anything"""
        self._kernel.train()

    def eval(self) -> None:
        """set eval mode, not actually evaluating anything"""
        self._kernel.eval()

    def get_hit_colours(self) -> int:
        return self._kernel.get_hit_colours()
    
    def get_missed_colours(self) -> int:
        return self._kernel.get_missed_colours()

    def fit(self, X, y) -> None:
        self._model.fit(X, y)

    def predict(self, X) -> np.array:
        return self._model.predict(X)

    def predict_with_std(self, X) -> Tuple[np.array, np.array]:
        """ for Bayesian models only """
        return self._model.predict(X, return_std=True)

    def get_learning_model(self):
        return self._model

    def lifted_state_input(self) -> bool:
        return self._representation.lifted

    def update_representation(self, domain_pddl: str, problem_pddl: str) -> None:
        self._representation: Representation = REPRESENTATIONS[self._rep_type](
            domain_pddl, problem_pddl
        )
        self._representation.convert_to_coloured_graph()
        return

    def get_iterations(self) -> int:
        return self._kernel.iterations

    def get_weight_indices(self):
        """Boolean array that is the size of self._model.coef_"""
        if hasattr(self, "_indices") and self._indices is not None:
            return self._indices
        return np.ones_like(self.get_weights())

    def set_weight_indices(self, indices):
        self._indices = indices
        return

    def get_weights(self):
        if self.model_name == "gp":
            # a hack: after training in train_bayes.py, use alpha @ X_train to get weights
            return self.weights
        
        weights = self._model.coef_
        if hasattr(self, "_indices") and self._indices is not None:
            weights = weights[self._indices]
        return weights

    def get_bias(self) -> float:
        if self.model_name == "gp":
            bias = 0
            return bias
        
        bias = self._model.intercept_
        if type(bias) == float:
            return bias
        if type(bias) == np.float64:
            return float(bias)
        return float(bias[0])  # linear-svr returns array

    def get_num_weights(self):
        return len(self.get_weights())

    def get_num_zero_weights(self):
        return np.count_nonzero(self.get_weights() == 0)

    def write_model_data(self) -> None:
        from datetime import datetime

        write_weights = self.model_name in LINEAR_MODELS

        df = self._representation.domain_pddl
        pf = self._representation.problem_pddl
        t = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_path = "_".join(["graph", df, pf, t])
        file_path = repr(hash(file_path)).replace("-", "n")
        file_path = file_path + ".model"

        model_hash = self.get_hash()
        iterations = self.get_iterations()

        if write_weights:
            weights = self.get_weights()
            indices = np.ones_like(weights)
            bias = self.get_bias()

            zero_weights = np.count_nonzero(weights == 0)
            print(f"{zero_weights}/{len(weights)} = {zero_weights/len(weights):.2f} are zero")

            # prune zero weights
            new_weights = []
            new_model_hash = {}

            reverse_hash = {model_hash[k]: k for k in model_hash}

            for colour, weight in enumerate(weights):
                if abs(weight) == 0 or indices[colour] == 0:
                    continue

                new_weights.append(weight)

                key = reverse_hash[colour]
                val = model_hash[key]
                new_model_hash[key] = val

            model_hash = new_model_hash
            weight = new_weights

        # write data
        with open(file_path, "w") as f:
            f.write(f"{NO_EDGE} NO_EDGE\n")
            f.write(f"{self.wl_name} wl_algorithm\n")
            f.write(f"{iterations} iterations\n")
            f.write(f"{len(model_hash)} hash size\n")
            for k in model_hash:
                f.write(f"{k} {model_hash[k]}\n")

            if write_weights:
                f.write(f"{len(weights)} weights size\n")
                for weight in weights:
                    f.write(str(weight) + "\n")
                f.write(f"{bias} bias\n")

        self._model_data_path = file_path
        pass

    def get_model_data_path(self) -> str:
        return self._model_data_path

    def write_representation_to_file(self) -> None:
        self._representation.write_to_file()
        return

    def get_graph_file_path(self) -> str:
        return self._representation.get_graph_file_path()

    def clear_graph(self) -> None:
        """Save memory for planner by deleting represntation once collected"""
        self._representation = None
        return

    def get_hash(self) -> Dict[str, int]:
        return self._kernel.get_hash()

    def compute_histograms(self, graphs: CGraph, return_ratio_seen_counts: bool = False) -> Union[List[Histogram], Tuple[List[Histogram], List[float]]]:
        return self._kernel.compute_histograms(graphs, return_ratio_seen_counts)

    def get_matrix_representation(
        self, graphs: CGraph, histograms: Optional[List[Histogram]]
    ) -> np.array:
        return self._kernel.get_x(graphs, histograms)

    def h(self, state: State) -> float:
        h = self.h_batch([state])[0]
        return h

    def h_batch(self, states: List[State]) -> List[float]:
        graphs = [self._representation.state_to_cgraph(state) for state in states]
        X = self._kernel.get_x(graphs)
        y = self.predict(X)
        hs = np.rint(y).astype(int).tolist()
        return hs

    def predict_h(self, x: Iterable[float]) -> float:
        """predict for single row x"""
        y = self.predict([x])
        return y
    
    def predict_h_with_std(self, x: Iterable[float]) -> Tuple[float, float]:
        y, std = self.predict_with_std([x])
        return (y, std)

    @property
    def n_colours_(self) -> int:
        return self._kernel.n_colours_
