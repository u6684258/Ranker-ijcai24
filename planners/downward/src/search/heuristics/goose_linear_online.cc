#include "goose_linear_online.h"

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

namespace goose_linear_online {

GooseLinearOnline::GooseLinearOnline(const plugins::Options &opts)
    : goose_linear::GooseLinear(opts) {
  train();
}

void GooseLinearOnline::train() {}

class GooseLinearOnlineFeature : public plugins::TypedFeature<Evaluator, GooseLinearOnline> {
 public:
  GooseLinearOnlineFeature() : TypedFeature("linear_model_online") {
    document_title("GOOSE optimised WL feature linear model heuristic with online training");
    document_synopsis("TODO");

    // https://github.com/aibasel/downward/pull/170 for string options
    add_option<std::string>("model_file", "path to trained model data in the form of a .model file",
                            "default_value");
    add_option<std::string>("graph_data", "path to trained model graph representation data",
                            "default_value");

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

static plugins::FeaturePlugin<GooseLinearOnlineFeature> _plugin;

}  // namespace goose_linear_online
