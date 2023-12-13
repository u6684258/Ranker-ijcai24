#ifndef HYPERGRAPH_HEURISTIC_H
#define HYPERGRAPH_HEURISTIC_H

#include "../heuristic.h"
#include <map>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace hypergraph_heuristic {

class HypergraphHeuristic : public Heuristic {
  // Builds up the mapping between Fast Downward fact names to pyperplan fact
  // names
  void initialize_pyperplan_strings();

  virtual int compute_heuristic(const GlobalState &global_state) override;

  // Creates the python heuristic object that will be called during search
  pybind11::object initialize_heuristic(const options::Options &opts) const;

  // Required for pybind. Once this goes out of scope python interaction is no
  // longer possible.
  //
  // IMPORTANT: since member variables are destroyed in reverse order of
  // construction it is important that the scoped_interpreter_guard is listed as
  // first member variable (therefore destroyed last), otherwise calling the
  // destructor of this class will lead to a segmentation fault.
  pybind11::scoped_interpreter guard;

  // Python object which computes the heuristic
  pybind11::object py_heuristic;

  // Dictionary that maps FD proposition strings to pyperplan proposition
  // strings. Since we have to pass strings to pyperplan these have to be
  // correct. We only want to do this translation once, hence we store it here.
  std::map<FactPair, std::string> fact_to_pyperstring;

public:
  explicit HypergraphHeuristic(const options::Options &opts);
  virtual ~HypergraphHeuristic() override;
};

} // namespace hypergraph_heuristic

#endif /* HYPERGRAPH_HEURISTIC_H */
