#ifndef GOOSE_LINEAR_ONLINE_H
#define GOOSE_LINEAR_ONLINE_H

#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../goose/coloured_graph.h"
#include "goose_linear.h"

/* Optimised linear regression model all in c++ with no pybind */

namespace goose_linear_online {

class GooseLinearOnline : public goose_linear::GooseLinear {
  void train();

 public:
  explicit GooseLinearOnline(const plugins::Options &opts);

 private:
  // A linear model of the form ax + b
  std::vector<double> weights_;  // a
  double bias_;                  // b
};

}  // namespace goose_linear_online

#endif  // GOOSE_LINEAR_ONLINE_H
