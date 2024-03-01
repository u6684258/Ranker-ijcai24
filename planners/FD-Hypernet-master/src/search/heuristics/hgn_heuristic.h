#ifndef HYPERGRAPH_HEURISTIC_H
#define HYPERGRAPH_HEURISTIC_H

#include "../heuristic.h"
#include <map>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace hgn_heuristic {

class HGNHeuristic : public Heuristic {
  // Builds up the mapping between Fast Downward fact names to pyperplan fact
  // names
  void initialize_pyperplan_strings();

  virtual int compute_heuristic(const GlobalState &global_state) override;

  // Required for pybind. Once this goes out of scope python interaction is no
  // longer possible.
  pybind11::scoped_interpreter guard;

  // Creates the python heuristic object that will be called during search
  pybind11::object initialize_heuristic(const options::Options &opts) const;

  // Computes the number of message passing steps M
  void compute_message_passing_steps(const options::Options &opts);

  // Python object which computes the heuristic
  pybind11::object py_heuristic;

  // Number of times the HGN core block is repeated. Corresponds to M in the
  // ICAPS paper by Shen et al.
  int num_steps;

  // Whether the network was trained on SAS problems or STRIPS problems
  bool use_sas_representation;

  // Dictionary that maps FD proposition strings to pyperplan proposition
  // strings. Since we have to pass strings to pyperplan these have to be
  // correct. We only want to do this translation once, hence we store it here.
  std::map<FactPair, std::string> fact_to_pyperstring;

public:
  explicit HGNHeuristic(const options::Options &opts);
  virtual ~HGNHeuristic() override = default;
};

} // namespace hgn_heuristic

#endif /* HYPERGRAPH_HEURISTIC_H */
