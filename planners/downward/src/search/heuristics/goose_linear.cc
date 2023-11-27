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
  lifted_goose = true;
  initialise_model(opts);
  initialise_lifted_facts();
}

void GooseLinear::initialise_model(const plugins::Options &opts) {
  std::string model_data_path = opts.get<string>("model_data");
  std::string graph_data_path = opts.get<string>("graph_data");

  std::cout << "Trying to load model data from files...\n";

  // load graph data
  graph_ = CGraph(graph_data_path);

  // load model data
  std::string line;
  std::ifstream infile(model_data_path);
  int hash_cnt = 0, hash_size = 0, weight_cnt = 0, weight_size = 0;

  // there's probably a better way to parse things
  while (std::getline(infile, line)) {
    std::vector<std::string> toks;
    std::istringstream iss(line);
    std::string s;
    while (std::getline(iss, s, ' ')) {
      toks.push_back(s);
    }
    if (line.find("hash size") != std::string::npos) {
      hash_size = stoi(toks[0]);
      hash_cnt = 0;
      continue;
    } else if (line.find("weights size") != std::string::npos) {
      weight_size = stoi(toks[0]);
      weight_cnt = 0;
      continue;
    } else if (line.find("bias") != std::string::npos) {
      bias_ = stod(toks[0]);
      continue;
    } else if (line.find("iterations") != std::string::npos) {
      iterations_ = stoi(toks[0]);
      continue;
    } else if (line.find("wl_algorithm") != std::string::npos) {
      wl_algorithm_ = toks[0];
      continue;
    } else if (line.find("NO_EDGE") != std::string::npos) {
      NO_EDGE_ = stoi(toks[0]);
      continue;
    }

    if (hash_cnt < hash_size) {
      hash_[toks[0]] = stoi(toks[1]);
      hash_cnt++;
      continue;
    }

    if (weight_cnt < weight_size) {
      weights_.push_back(stod(line));
      weight_cnt++;
      continue;
    }
  }

  // remove file
  char *char_array = new char[model_data_path.length() + 1];
  strcpy(char_array, model_data_path.c_str());
  remove(char_array);

  feature_size_ = static_cast<int>(weights_.size());
}

int GooseLinear::predict(const std::vector<int> &feature) {
  double ret = bias_;
  for (int i = 0; i < feature_size_; i++) {
    ret += feature[i] * weights_[i];
  }
  return static_cast<int>(round(ret));
}

int GooseLinear::compute_heuristic(const State &ancestor_state) {
  // int cur_seen_colours = cnt_seen_colours;
  // int cur_unseen_colours = cnt_unseen_colours;

  // step 1.
  CGraph graph = state_to_graph(ancestor_state);
  // step 2.
  std::vector<int> feature = wl_feature(graph);
  // step 3.
  int h = predict(feature);

  // cur_seen_colours = cnt_seen_colours - cur_seen_colours;
  // cur_unseen_colours = cnt_unseen_colours - cur_unseen_colours;
  // double ratio_seen_colours = static_cast<double>(cur_seen_colours) / cur_unseen_colours;

  // worst_seen_ratios.insert(ratio_seen_colours);

  // if (worst_seen_ratios.size() == 6) {
  //   worst_seen_ratios.erase(std::prev(worst_seen_ratios.end()));
  // }
  // if (worst_seen_ratios.count(ratio_seen_colours)) {
  //   std::cout << h << " " << ratio_seen_colours << std::endl;
  //   for (const auto r :worst_seen_ratios) {
  //     std::cout<<" "<<r;
  //   }
  //   std::cout<<std::endl;
  // }

  // if (!h_to_worst_ratio.count(h)) {
  //   h_to_worst_ratio[h] = ratio_seen_colours;
  // }
  // h_to_worst_ratio[h] = std::min(ratio_seen_colours, h_to_worst_ratio[h]);

  return h;
}

void GooseLinear::print_statistics() const {
  log << "Number of seen " << wl_algorithm_ << " colours: " << cnt_seen_colours << std::endl;
  log << "Number of unseen " << wl_algorithm_ << " colours: " << cnt_unseen_colours << std::endl;
  // for (auto const [h, r] : h_to_worst_ratio) {
  //   std::cout<<h<<" " <<r<< std::endl;
  // }
}

class GooseLinearFeature : public plugins::TypedFeature<Evaluator, GooseLinear> {
 public:
  GooseLinearFeature() : TypedFeature("linear_model") {
    document_title("GOOSE optimised WL feature linear model heuristic");
    document_synopsis("TODO");

    // https://github.com/aibasel/downward/pull/170 for string options
    add_option<std::string>("model_data", "path to trained model data in the form of a .model file",
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

static plugins::FeaturePlugin<GooseLinearFeature> _plugin;

}  // namespace goose_linear
