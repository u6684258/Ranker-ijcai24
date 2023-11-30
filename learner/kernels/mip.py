import numpy as np
import time
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted


class MIP(BaseEstimator):
    def __init__(self):
        import pulp
        assert pulp.apis.CPLEX_PY().available()
        self.coef_ = np.array([0])
        self.bias_ = 0  # no bias
        self.intercept_ = 0  # no bias
        pass

    def fit(self, X, y):
        import pulp
        from pulp import LpVariable as Var
        from pulp import lpDot, lpSum
        t = time.time()
        print("Constructing MIP problem...")
        m = pulp.LpProblem()
        n, d = X.shape
        n -= 1

        tiebreaker = X[-1]  # see KernelModelWrapper._transform_for_fit_only

        """ Variables """
        print("Constructing variables...")
        weights = [
            Var(f"w:{j}", cat=pulp.const.LpInteger, lowBound=-1, upBound=1) for j in range(d)
        ]
        weights_abs = [Var(f"w_abs:{j}") for j in range(d)]
        diffs = [Var(f"diff:{i}") for i in range(n)]

        """ Objective and Constraints """
        print("Constructing objective and constraints...")
        # minimise L1 distance loss
        for i in range(n):
            pred = lpDot(weights, X[i])
            m += diffs[i] >= pred - y[i]
            m += diffs[i] >= y[i] - pred
        main_obj = lpSum(diffs)  # abs value of differences

        # minimise weights for tie breaking
        for j in range(d):
            m += weights_abs[j] >= -weights[j]
            m += weights_abs[j] >= weights[j]
            
        m += sum(y) * main_obj + lpDot(weights_abs, tiebreaker)
        # m += sum(y) * main_obj + lpSum(weights_abs)
        
        print(f"MIP problem constructed in {time.time() - t:.2f}s!")

        # TODO warm start

        """ Solve """
        m.checkDuplicateVars()
        solver = pulp.getSolver("CPLEX_PY", timeLimit=10)  # TODO argument for time limit
        m.solve(solver)

        self.coef_ = np.array([w.value() for w in weights])

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        y = X @ self.coef_
        return y
