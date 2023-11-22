#ifndef GOOSE_HEURISTIC_H
#define GOOSE_HEURISTIC_H

#include <map>
#include "../heuristic.h"

namespace goose_heuristic {

class GooseHeuristic : public Heuristic {
 protected:
  void initialise_grounded_facts();
  void initialise_lifted_facts();
  void initialise_facts();

  // Dictionary that maps FD proposition strings to (pred o_1 ... o_n)
  // proposition strings in the case of grounded GOOSE, or (pred, args) tuples.
  // We only want to do this translation once,
  // hence we store it here. This could be ignored if we change the format of
  // propositions in GOOSE.
  std::map<FactPair, std::string> fact_to_grounded_input;
  std::map<FactPair, std::pair<std::string, std::vector<std::string>>> fact_to_lifted_input;

  bool lifted_goose;

 public:
  explicit GooseHeuristic(const plugins::Options &opts);
};

}  // namespace goose_heuristic

#endif
