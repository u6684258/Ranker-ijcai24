#ifndef GOOSE_LINEAR_ONLINE_H
#define GOOSE_LINEAR_ONLINE_H

#include "../goose/coloured_graph.h"
#include "goose_linear.h"

/* Performs online learning by regressing from goal condition and using seen colours ratios to
decide what states to train on */

namespace goose_linear_online {

typedef std::vector<int> PartialState;
typedef std::vector<FactPair> FullState;

struct SearchNodeStats {
  int h;
  std::vector<double> ratio;

  SearchNodeStats(int h, const std::vector<double> ratio) : h(h), ratio(ratio){};
};

struct BackwardsSearchNode {
  PartialState state;
  int y;
  // int h;
  // std::vector<double> ratio;

  BackwardsSearchNode(const PartialState &state, int y)
      : state(state), y(y) {};

  // BackwardsSearchNode(const PartialState &state, int y, const SearchNodeStats &stats)
  //     : state(state), h(stats.h), y(y), ratio(stats.ratio){};
};

class GooseLinearOnline : public goose_linear::GooseLinear {
  // does not account for mutexes
  FullState assign_random_state(const PartialState &state);

  void train();

  SearchNodeStats compute_heuristic_vector_state(const FullState &state);

 public:
  explicit GooseLinearOnline(const plugins::Options &opts);

 private:
  int n_variables;
  std::random_device dev;
  std::mt19937 rng;
  VariablesProxy vars;
};

}  // namespace goose_linear_online

#endif  // GOOSE_LINEAR_ONLINE_H
