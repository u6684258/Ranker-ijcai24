#ifndef GOOSE_WL_HEURISTIC_H
#define GOOSE_WL_HEURISTIC_H

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../goose/coloured_graph.h"
#include "goose_heuristic.h"

/* class extending GooseHeuristic for performing all WL computations
    could have been made a decoupled class that does not need to extend GooseHeuristic..
*/

namespace goose_wl {

class WLGooseHeuristic : public goose_heuristic::GooseHeuristic {
 protected:
  // convert state to CGraph (ILG representation)
  CGraph state_to_graph(const State &state);

  // perform 1-WL on CGraph and return a feature vector of colour counts
  std::vector<int> wl_feature(const CGraph &graph);

 public:
  explicit WLGooseHeuristic(const plugins::Options &opts);

 protected:
  CGraph graph_;
  std::map<std::string, int> hash_;
  int feature_size_;  // size of hash and feature vectors = number of unique training colours
  size_t iterations_;  // number of WL iterations
};

}  // namespace goose_wl

#endif  // GOOSE_WL_HEURISTIC_H
