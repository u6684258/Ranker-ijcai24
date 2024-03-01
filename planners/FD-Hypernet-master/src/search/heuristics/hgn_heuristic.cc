#include "hgn_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"

#include "ff_heuristic.h"

#include <algorithm>
#include <cctype>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <set>

#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace fs = std::experimental::filesystem;

using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace hgn_heuristic {

HGNHeuristic::HGNHeuristic(const Options &opts) : Heuristic(opts) {
  cout << "Initializing new hypergraph heuristic" << endl;

  use_sas_representation = opts.get<bool>("sas");
  num_steps = opts.get<int>("num_steps");
  if (num_steps == 0) {
    compute_message_passing_steps(opts);
  }
  cout << "Num Steps: " << num_steps << endl;
  py_heuristic = initialize_heuristic(opts);
  if (!use_sas_representation) {
    initialize_pyperplan_strings();
  }
}

void HGNHeuristic::compute_message_passing_steps(const Options &opts) {
  ff_heuristic::FFHeuristic hff(opts);
  num_steps = hff.compute_heuristic(task_proxy.get_initial_state());
}

void HGNHeuristic::initialize_pyperplan_strings() {
  FactsProxy facts(*task);
  for (FactProxy fact : facts) {
    string name = fact.get_name().substr(5);
//    cout << name << " ";
    // replace all occurences of '(' and ')' by ' '
    std::replace(name.begin(), name.end(), '(', ' ');
    std::replace(name.begin(), name.end(), ')', ' ');
    // Remove occurences of ','
    name.erase(std::remove(name.begin(), name.end(), ','), name.end());
    // Trim string
    if (std::isspace(name[0])) {
      name.erase(0, 1);
    }
    if (std::isspace(name.back())) {
      name.erase(name.end() - 1, name.end());
    }
    // Add parantheses around string
    name = "(" + name + ")";
    fact_to_pyperstring.insert({fact.get_pair(), name});
  }
  cout << endl;
}

py::object HGNHeuristic::initialize_heuristic(const Options &opts) const {
  // Add learning heuristics submodule to the python path
  auto hgn_path = std::getenv("STRIPS_HGN_NEW");
  if (!hgn_path) {
    cout << "STRIPS_HGN_NEW environment variable not found. Aborting." << endl;
    utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
  }
  string path(hgn_path);
  cout << "Strips HGN path is " << path << endl;
  if (!fs::exists(hgn_path)) {
    cout << "STRIPS_HGN_NEW points to non-existent path. Aborting." << endl;
    utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
  }

  // Append python module directory to the path
  py::module sys = py::module::import("sys");
  sys.attr("path").attr("append")(path);
  sys.attr("path").attr("append")(path + "/src/pyperplan/");

  // Force all output being printed to stdout. Otherwise INFO logging from
  // python will be printed to stderr, even if it is not an error.
  sys.attr("stderr") = sys.attr("stdout");

  // Append heuristic path to python path
  auto fd_path = std::getenv("FD_HGN");
  if (!fd_path) {
    cout << "FD_HGN environment variable not found. Aborting." << endl;
    utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
  }
  string heur_path = static_cast<string>(fd_path) + "/src/search/heuristics";
  sys.attr("path").attr("append")(heur_path);
  cout << "state evaluation path is " << heur_path << endl;

  // Output network file location
  string network_file = opts.get<string>("network_file");
  cout << "Loading network from " << network_file << endl;

  bool multithreading = opts.get<bool>("multithreading");

  cout << "Using multiple cores: " << std::to_string(multithreading) << endl;

  // Construct hypergraph view from strips problem
  string domain_file = opts.get<string>("domain_file");
  string instance_file = opts.get<string>("instance_file");

  cout << "Instantiating HGN evaluation object" << endl;
  py::module eval_module = py::module::import("state_evaluation");
  py::object heuristic =
      eval_module.attr("HGNEvaluator")(domain_file, instance_file, network_file,
                                       multithreading, use_sas_representation, opts.get<string>("type"));
  cout << "Finished setup of HGN heuristic" << endl;

  return heuristic;
}

int HGNHeuristic::compute_heuristic(const GlobalState &global_state) {
  const State state = convert_global_state(global_state);
  py::set py_facts;
  if (use_sas_representation) {
    vector<int> const &values = state.get_values();
    // If we use SAS representation a HGN state is simply a set of
    // variable-value pairs
    for (size_t var = 0; var < values.size(); ++var) {
      py_facts.add(py::make_tuple(var, values[var]));
    }
  } else {
    for (FactProxy fact : state) {
      // In python HGN a state is a (frozen) set of strings, each string
      // corresponding to a fact and is preprended by "Atom" which we remove
      // here
      py_facts.add(fact_to_pyperstring[fact.get_pair()]);
    }
  }
  // py::print(py_facts);
  py::object value = py_heuristic.attr("evaluate_state")(py_facts, num_steps);
  int result = static_cast<int>(value.cast<float>());
//  cout << result << endl;
  return result;
}

static std::shared_ptr<Heuristic> _parse(OptionParser &parser) {
  parser.document_synopsis("Hypergraph networks heuristic", "");
  parser.document_language_support("action costs", "not supported");
  parser.document_language_support("conditional effects", "not supported");
  parser.document_language_support("axioms", "not supported");
  parser.document_property("admissible", "no");
  parser.document_property("consistent", "no");
  parser.document_property("safe", "yes");
  parser.document_property("preferred operators", "no");

  Heuristic::add_options_to_parser(parser);
  parser.add_option<string>(
      "network_file",
      "Path where the neural network is stored that will be used for the "
      "heuristic. Note that the path MUST NOT contain upper case letters.",
      OptionParser::NONE);

  parser.add_option<string>("domain_file",
                            "Path to the domain file. The path MUST NOT "
                            "contain upper case letters.",
                            OptionParser::NONE);

  parser.add_option<string>("instance_file",
                            "Path to the instance file. The path MUST NOT "
                            "contain upper case letters.",
                            OptionParser::NONE);

  parser.add_option<bool>("multithreading",
                          "Allows pytorch to use multiple cores.", "false");

  parser.add_option<int>(
      "num_steps",
      "Number of times the HGN core block is repeated. Pass 0 to use hff to "
      "automatically determine a step number.",
      "0", Bounds("0", "50"));

  parser.add_option<bool>(
      "sas", "Use if network was trained on SAS description.", "false");

  parser.add_option<string>(
      "type", "Use if network was trained on SAS description.", "hgn");

  Options opts = parser.parse();
  if (parser.help_mode()) {
    return nullptr;
  }
  if (parser.dry_run())
    return nullptr;
  else
    return std::make_shared<HGNHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("hgn2", _parse);

} // namespace hgn_heuristic

