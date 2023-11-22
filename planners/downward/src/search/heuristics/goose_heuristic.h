#ifndef GOOSE_HEURISTIC_H
#define GOOSE_HEURISTIC_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../heuristic.h"

namespace goose_heuristic {

struct FactPairHasher {
  size_t operator()(const FactPair &a) const { return std::hash<int>{}(31 * a.var + a.value); }
};

struct FactPairEquals {
  bool operator()(const FactPair &a, const FactPair &b) const {
    return 31 * a.var + a.value == 31 * b.var + b.value;
  }
};

typedef std::string GroundedInput;
typedef std::pair<std::string, std::vector<std::string>> LiftedInput;

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
  std::map<FactPair, GroundedInput> fact_to_g_input;
  std::map<FactPair, LiftedInput> fact_to_l_input;
  // std::unordered_map<FactPair, GroundedInput, FactPairHasher, FactPairEquals> fact_to_g_input;
  // std::unordered_map<FactPair, LiftedInput, FactPairHasher, FactPairEquals> fact_to_l_input;

  bool lifted_goose;

 public:
  explicit GooseHeuristic(const plugins::Options &opts);
};

}  // namespace goose_heuristic

#endif
