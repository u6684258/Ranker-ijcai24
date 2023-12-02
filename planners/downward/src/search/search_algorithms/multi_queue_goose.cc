#include "multi_queue_goose.h"

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../open_list_factory.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../plugins/options.h"
#include "../task_utils/successor_generator.h"
#include "../utils/logging.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>

using namespace std;

namespace multi_queue_goose {
MultiQueueGoose::MultiQueueGoose(const plugins::Options &opts) : SearchAlgorithm(opts) {
  std::shared_ptr<Evaluator> eval = opts.get_list<std::shared_ptr<Evaluator>>("evals")[0];
  goose_heuristic = dynamic_pointer_cast<goose_linear::GooseLinear>(eval);
  n_linear_models_ = goose_heuristic->num_linear_models();
  for (int i = 0; i < n_linear_models_; i++) {
    open_lists.push_back(GooseOpenList<StateID>());
  }
}

void MultiQueueGoose::initialize() {
  log << "Conducting best first search without"
      << " reopening closed nodes, (real) bound = " << bound << endl;

  // hack: just do everything here

  // TODO(DZC) more verbose loging: best computed h for each heuristic
  int q_cnt = 0;

  State initial_state = state_registry.get_initial_state();
  std::vector<int> feature = goose_heuristic->get_feature(initial_state);
  for (int i = 0; i < n_linear_models_; i++) {
    int h = goose_heuristic->compute_heuristic_from_feature(feature, i);
    open_lists[i].insert(h, initial_state.get_id());
  }

  // just assume solvable problems
  while (true) {
    optional<SearchNode> node;
    StateID s_id = open_lists[q_cnt].remove_min();
    q_cnt = (q_cnt + 1) % n_linear_models_;
    State s = state_registry.lookup_state(s_id);
    node.emplace(search_space.get_node(s));

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(s, applicable_ops);
    statistics.inc_expanded();

    for (OperatorID op_id : applicable_ops) {
      OperatorProxy op = task_proxy.get_operators()[op_id];

      State succ_state = state_registry.get_successor_state(s, op);
      statistics.inc_generated();
      SearchNode succ_node = search_space.get_node(succ_state);

      if (check_goal_and_set_plan(succ_state)) {
        log << "OK" << std::endl;
        return;
      }

      if (succ_node.is_new()) {
        statistics.inc_evaluated_states(n_linear_models_);
        succ_node.open(*node, op, get_adjusted_cost(op));

        feature = goose_heuristic->get_feature(succ_state);
        for (int i = 0; i < n_linear_models_; i++) {
          int h = goose_heuristic->compute_heuristic_from_feature(feature, i);
          open_lists[i].insert(h, succ_state.get_id());
        }
      }
    }
  }
}

void MultiQueueGoose::print_statistics() const {
  statistics.print_detailed_statistics();
  search_space.print_statistics();
}

SearchStatus MultiQueueGoose::step() { return SOLVED; }

void MultiQueueGoose::dump_search_space() const { search_space.dump(task_proxy); }

void add_options_to_feature(plugins::Feature &feature) {
  SearchAlgorithm::add_pruning_option(feature);
  SearchAlgorithm::add_options_to_feature(feature);
}
}  // namespace multi_queue_goose
