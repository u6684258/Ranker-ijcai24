#include "goose_kernel.h"

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

namespace goose_kernel {

GooseKernel::GooseKernel(const plugins::Options &opts)
    : goose_wl::WLGooseHeuristic(opts) {
  initialise_model(opts);
  initialise_lifted_facts();
}

void GooseKernel::initialise_model(const plugins::Options &opts) {
  // Add GOOSE submodule to the python path
  auto gnn_path = std::getenv("GOOSE");
  if (!gnn_path) {
    std::cout << "GOOSE env variable not found. Aborting." << std::endl;
    exit(-1);
  }
  std::string path(gnn_path);
  std::cout << "GOOSE path is " << path << std::endl;
  if (access(path.c_str(), F_OK) == -1) {
    std::cout << "GOOSE points to non-existent path. Aborting." << std::endl;
    exit(-1);
  }

  // Append python module directory to the path
  pybind11::module sys = pybind11::module::import("sys");
  sys.attr("path").attr("append")(path);

  // Force all output being printed to stdout. Otherwise INFO logging from
  // python will be printed to stderr, even if it is not an error.
  sys.attr("stderr") = sys.attr("stdout");

  std::string model_path = opts.get<std::string>("model_data");
  std::string domain_file = opts.get<std::string>("domain_file");
  std::string instance_file = opts.get<std::string>("instance_file");
  std::cout << "Trying to load model from file " << model_path << " ...\n";
  pybind11::module util_module = pybind11::module::import("util.save_load");
  model = util_module.attr("load_kernel_model_and_setup")(model_path, domain_file, instance_file);
  std::cout << "Loaded model!" << std::endl;

  // use I/O similar to goose_linear_regression to get graph representation and WL data
  model.attr("write_model_data")(0);
  model.attr("write_representation_to_file")();
  std::string model_data_path = model.attr("get_model_data_path")().cast<std::string>();
  std::string graph_data_path = model.attr("get_graph_file_path")().cast<std::string>();
  graph_ = CGraph(graph_data_path);

  // load WL hash data
  std::string line;
  std::ifstream infile(model_data_path);
  int hash_cnt = 0, hash_size = 0;
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
      feature_size_ = hash_size;
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
  }

  // remove file
  char *char_array = new char[model_data_path.length() + 1];
  strcpy(char_array, model_data_path.c_str());
  remove(char_array);

  std::cout << "Model initialised!" << std::endl;
}

int GooseKernel::predict(const std::vector<int> &feature) {
  int h = model.attr("svr_predict")(feature).cast<int>();
  return h;
}

int GooseKernel::compute_heuristic(const State &ancestor_state) {
  // step 1.
  CGraph graph = state_to_graph(ancestor_state);
  // step 2.
  std::vector<int> feature = wl_feature(graph);
  // step 3.
  int h = predict(feature);
  return h;
}

class GooseKernelFeature : public plugins::TypedFeature<Evaluator, GooseKernel> {
 public:
  GooseKernelFeature() : TypedFeature("kernel_model") {
    document_title("GOOSE optimised WL kernel heuristic");
    document_synopsis("TODO");

    // https://github.com/aibasel/downward/pull/170 for string options
    add_option<std::string>(
        "model_data", "path to trained model data in the form of a .joblib file", "default_value");
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

static plugins::FeaturePlugin<GooseKernelFeature> _plugin;

}  // namespace goose_kernel
