#include "hypergraph_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../utils/system.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
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

namespace hypergraph_heuristic {

// Enums for option arguments

enum class PaddingOption { ZERO, REPEAT };

enum class EdgeMapperOption { WEIGHT_ONLY, COMPLEX };

HypergraphHeuristic::HypergraphHeuristic(const Options &opts)
    : Heuristic(opts) {
  // TODO pass arguments from options
  cout << "Initializing hypergraph heuristic" << endl;

  py_heuristic = initialize_heuristic(opts);
  initialize_pyperplan_strings();
}

void HypergraphHeuristic::initialize_pyperplan_strings() {
  FactsProxy facts(*task);
  for (FactProxy fact : facts) {
    string name = fact.get_name().substr(5);
    cout << name << " ";
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

HypergraphHeuristic::~HypergraphHeuristic() {}

py::object
HypergraphHeuristic::initialize_heuristic(const Options &opts) const {
  // Add learning heuristics submodule to the python path
  auto hgn_path = std::getenv("STRIPS_HGN");
  if (!hgn_path) {
    cout << "STRIPS_HGN environment variable not found. Aborting." << endl;
    utils::exit_with(utils::ExitCode::SEARCH_UNSUPPORTED);
  }

  cout << "Strips HGN path is " << hgn_path << endl;
  // Append python module directory to the path
  string path(hgn_path);
  path += "/learning_heuristics/";
  py::module sys = py::module::import("sys");
  sys.attr("path").attr("append")(path);
  py::module pyperplan_module =
      py::module::import("learning_heuristics.pyperplan_adaptors");
  py::module feature_mapper_module =
      py::module::import("learning_heuristics.feature_mappers");
  py::module adaptor_module =
      py::module::import("learning_heuristics.hypergraph_nets_adaptor");

  string network_file = opts.get<string>("network_file");
  cout << "Loading network from " << network_file << endl;

  // neural network
  py::module torch = py::module::import("torch");
  py::object model = torch.attr("load")(network_file);

  // hypergraph
  string domain_file = opts.get<string>("domain_file");
  string instance_file = opts.get<string>("instance_file");
  py::object hg_class = pyperplan_module.attr("DeleteRelaxationHypergraph");
  py::object hypergraph =
      hg_class.attr("extract_hypergraph")(domain_file, instance_file);

  // feature mappers
  py::object node_mapper =
      feature_mapper_module.attr("simple_node_feature_mapper");
  auto feature_mapper =
      static_cast<EdgeMapperOption>(opts.get_enum("edge_feature_mapper"));

  py::object edge_mapper =
      (feature_mapper == EdgeMapperOption::COMPLEX)
          ? feature_mapper_module.attr("complex_edge_feature_mapper")
          : feature_mapper_module.attr("weight_only_edge_feature_mapper");

  auto padding_option =
      static_cast<PaddingOption>(opts.get_enum("pad_function"));
  py::object padding = (padding_option == PaddingOption::ZERO)
                           ? adaptor_module.attr("pad_with_obj_up_to_k")
                           : adaptor_module.attr("repeat_up_to_k");

  int num_steps = opts.get<int>("num_steps");
  cout << "Num Steps: " << num_steps << endl;

  // Compute network receiver and sender size. Receiver size is the maximum
  // number of add effects of any operator and sender is the maximum number of
  // preconditions of any operator
  size_t receivers = 0;
  size_t senders = 0;
  OperatorsProxy operators(*task);
  for (OperatorProxy op : operators) {
    // Since we currently require the translator to keep the full encoding some
    // operators have potentially multiple copies of the same fact as
    // precondition, since one fact can occur in multiple variables. We filter
    // out copies of preconditions here.
    std::set<string> preconditions;
    for (FactProxy fact : op.get_preconditions()) {
      preconditions.insert(fact.get_name());
    }
    std::set<string> effects;
    for (EffectProxy effect : op.get_effects()) {
      effects.insert(effect.get_fact().get_name());
    }
    senders = std::max(senders, preconditions.size());
    receivers = std::max(receivers, effects.size());
  }
  cout << receivers << " receivers and " << senders << " senders." << endl;

  // heuristic object
  py::object py_heuristic = pyperplan_module.attr("HypergraphNetsHeuristic")(
      model, hypergraph, num_steps, node_mapper, edge_mapper, padding,
      receivers, senders);
  return py_heuristic;
}

int HypergraphHeuristic::compute_heuristic(const GlobalState &global_state) {
  const State state = convert_global_state(global_state);
  py::set py_facts;
  // In python HGN a state is a (frozen) set of strings, each string
  // corresponding to a fact and is preprended by "Atom" which we remove here
  for (FactProxy fact : state) {
    py_facts.add(fact_to_pyperstring[fact.get_pair()]);
  }

  // TODO correct name mapping between FD and pyperplan
  py::object value = py_heuristic.attr("evaluate_state")(py_facts);
  // py::print(value);
  int result = static_cast<int>(value.cast<float>());
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

  parser.add_option<string>(
      "domain_file",
      "Path to the domain file. The path MUST NOT contain upper case letters.",
      OptionParser::NONE);

  parser.add_option<string>("instance_file",
                            "Path to the instance file. The path MUST NOT "
                            "contain upper case letters.",
                            OptionParser::NONE);

  vector<string> argument_strings = {"weight-only", "complex"};
  vector<string> argument_explanation = {"TBD: explanation for weight-only",
                                         "TBD: explanation for complex"};
  parser.add_enum_option("edge_feature_mapper", argument_strings,
                         "How a hyperedges is mapped to its feature vector."
                         "will be used for the heuristic.",
                         "weight-only", argument_explanation);

  parser.add_option<int>("num_steps", "TBD: explanation for num_steps.", "10",
                         Bounds("1", "1000"));

  argument_strings = {"zero", " repeat"};
  argument_explanation = {"TBD: explanation for zero padding.",
                          "TBD: explanation for repeated padding"};
  parser.add_enum_option("pad_function", argument_strings,
                         "TBD: explanation for padding functions.", "zero",
                         argument_explanation);

  // receiver_k : max add effects over all operators
  // receiver_k : max preconditions over all operators

  Options opts = parser.parse();
  if (parser.help_mode()) {
    return nullptr;
  }
  if (parser.dry_run())
    return nullptr;
  else
    return std::make_shared<HypergraphHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("hgn", _parse);

} // namespace hypergraph_heuristic

