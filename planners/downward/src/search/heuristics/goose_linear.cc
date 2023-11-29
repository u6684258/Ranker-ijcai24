#include "goose_linear.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"

using std::string;

namespace goose_linear {

GooseLinear::GooseLinear(const plugins::Options &opts) : goose_wl::WLGooseHeuristic(opts) {
  model = pybind11::int_(0);  // release memory since we no longer need the python object
}

int GooseLinear::predict(const std::vector<int> &feature) {
  double ret = bias_;
  for (int i = 0; i < feature_size_; i++) {
    ret += feature[i] * weights_[i];
  }
  return static_cast<int>(round(ret));
}

int GooseLinear::compute_heuristic(const State &ancestor_state) {
  // step 1.
  CGraph graph = state_to_graph(ancestor_state);
  // step 2.
  std::vector<int> feature = wl_feature(graph);
  // step 3.
  int h = predict(feature);

  return h;
}

class GooseLinearFeature : public plugins::TypedFeature<Evaluator, GooseLinear> {
 public:
  GooseLinearFeature() : TypedFeature("linear_model") {
    document_title("GOOSE optimised WL feature linear model heuristic");
    document_synopsis("TODO");

    // https://github.com/aibasel/downward/pull/170 for string options
    add_option<std::string>("model_file", "path to trained python model", "default_value");
    add_option<std::string>("domain_file", "Path to the domain file.", "default_file");
    add_option<std::string>("instance_file", "Path to the instance file.", "default_file");

    Heuristic::add_options_to_feature(*this);

    document_language_support("action costs", "not supported");
    document_language_support("conditional effects", "not supported");
    document_language_support("axioms", "not supported");

    document_property("admissible", "no");
    document_property("consistent", "no");
    document_property("safe", "yes");
    document_property("preferred operators", "no");
  }
};

static plugins::FeaturePlugin<GooseLinearFeature> _plugin;

}  // namespace goose_linear
