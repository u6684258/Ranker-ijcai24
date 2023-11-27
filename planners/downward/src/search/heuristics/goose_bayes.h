#ifndef GOOSE_BAYES_H
#define GOOSE_BAYES_H

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>

#include <map>
#include <set>
#include <vector>
#include <utility>
#include <string>
#include <fstream>

#include "goose_wl_heuristic.h"
#include "../goose/coloured_graph.h"


/* Bayes model which calls python sklearn for evaluation */

namespace goose_bayes {

class GooseBayes : public goose_wl::WLGooseHeuristic {
  void initialise_model(const plugins::Options &opts);

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

  /* Heuristic computation consists of three steps */

  // 1. convert state to CGraph (IG representation)
  // see goose_wl::WLGooseHeuristic
  // 2. perform WL on CGraph
  // see goose_wl::WLGooseHeuristic
  // 3. make a prediction with explicit feature
  // see compute_heuristic

 protected:
  int compute_heuristic(const State &ancestor_state) override;
  
 public:
  explicit GooseBayes(const plugins::Options &opts);

  void print_statistics() const override;

//  private:
//   std::set<std::tuple<double, double, double>> std_ratio_pairs;
};

}  // namespace goose_bayes

#endif  // GOOSE_BAYES_H
