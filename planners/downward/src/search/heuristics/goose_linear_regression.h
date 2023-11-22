#ifndef GOOSE_LINEAR_REGRESSION_H
#define GOOSE_LINEAR_REGRESSION_H

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../goose/coloured_graph.h"
#include "goose_wl_heuristic.h"

/* Optimised linear regression model all in c++ with no pybind */

namespace goose_linear_regression {

class GooseLinearRegression : public goose_wl::WLGooseHeuristic {
  void initialise_model(const plugins::Options &opts);

  /* Heuristic computation consists of three steps */

  // 1. convert state to CGraph (IG representation)
  // see goose_wl::WLGooseHeuristic
  // 2. perform WL on CGraph
  // see goose_wl::WLGooseHeuristic
  // 3. make a prediction with explicit feature
  int predict(const std::vector<int> &feature);

 protected:
  int compute_heuristic(const State &ancestor_state) override;

 public:
  explicit GooseLinearRegression(const plugins::Options &opts);

 private:
  // A linear model of the form ax + b
  std::vector<double> weights_;  // a
  double bias_;                  // b
};

}  // namespace goose_linear_regression

#endif  // GOOSE_LINEAR_REGRESSION_H
