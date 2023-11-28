#ifndef GOOSE_WL_HEURISTIC_H
#define GOOSE_WL_HEURISTIC_H

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>

#include <fstream>
#include <map>
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
  // Required for pybind. Once this goes out of scope python interaction is no
  // longer possible.
  //
  // IMPORTANT: since member variables are destroyed in reverse order of
  // construction it is important that the scoped_interpreter_guard is listed as
  // first member variable (therefore destroyed last), otherwise calling the
  // destructor of this class will lead to a segmentation fault.
  pybind11::scoped_interpreter guard;

  // Python object which computes the heuristic
  pybind11::object model;

  // convert state to CGraph (ILG representation)
  CGraph fact_pairs_to_graph(const std::vector<FactPair> &state);
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
  std::vector<long> cnt_seen_colours;
  std::vector<long> cnt_unseen_colours;

  // the following variables are pretty much constant after initialisation
  std::string wl_algorithm_;
  CGraph graph_;
  std::unordered_map<std::string, int> hash_;
  int feature_size_;   // size of hash and feature vectors = number of unique training colours
  size_t iterations_;  // number of WL iterations

  // For linear models of the form ax + b
  std::vector<double> weights_;  // a
  double bias_;                  // b

  // represents no edge for 2-wl methods, read from file but treat it as a constant
  int NO_EDGE_;  
};

}  // namespace goose_wl

#endif  // GOOSE_WL_HEURISTIC_H
