#ifndef GOOSE_WL_HEURISTIC_H
#define GOOSE_WL_HEURISTIC_H

#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
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

  // perform the wl algorithm specified by wl_algorithm_ to
  // return a feature vector of colour counts
  std::vector<int> wl_feature(const CGraph &graph);

  std::vector<int> wl1_feature(const CGraph &graph);
  std::vector<int> gwl2_feature(const CGraph &graph);
  std::vector<int> lwl2_feature(const CGraph &graph);
  std::vector<int> lwl3_feature(const CGraph &graph);

 public:
  explicit WLGooseHeuristic(const plugins::Options &opts);

  void print_statistics() const override;

 protected:
  // counters to keep track of number of seen and unseen colours
  // hopefully should not overflow (max val=9,223,372,036,854,775,807)
  long cnt_seen_colours;
  long cnt_unseen_colours;

  // the following variables are pretty much constant after initialisation
  std::string wl_algorithm_;
  CGraph graph_;
  std::unordered_map<std::string, int> hash_;
  int feature_size_;   // size of hash and feature vectors = number of unique training colours
  size_t iterations_;  // number of WL iterations

  int NO_EDGE_;  // represents no edge for 2-wl methods, read from file but treat it as a constant
};

}  // namespace goose_wl

#endif  // GOOSE_WL_HEURISTIC_H
