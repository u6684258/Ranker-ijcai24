#ifndef GOOSE_LINEAR_H
#define GOOSE_LINEAR_H

#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../goose/coloured_graph.h"
#include "goose_wl_heuristic.h"

/* Optimised linear regression model all in c++ with no pybind */

namespace goose_linear {

class GooseLinear : public goose_wl::WLGooseHeuristic {
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
  explicit GooseLinear(const plugins::Options &opts);

  void print_statistics() const override;

 private:
  // A linear model of the form ax + b
  std::vector<double> weights_;  // a
  double bias_;                  // b

  // // colour seen ratio
  // std::set<double> worst_seen_ratios;
  // std::map<int, double> h_to_worst_ratio;
};

}  // namespace goose_linear

#endif  // GOOSE_LINEAR_H
