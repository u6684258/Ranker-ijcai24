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
  symmetry_ = opts.get<bool>("symmetry");
  for (int i = 0; i < n_linear_models_; i++) {
    open_lists.push_back(GooseOpenList<StateID>());
  }
}

void MultiQueueGoose::initialize() {
  log << "Conducting best first search without"
      << " reopening closed nodes, (real) bound = " << bound << endl;

  // hack: just do everything here

  std::vector<int> best_h(n_linear_models_, bound);

  int q_cnt = 0;  // multi queue cycler count

  State initial_state = state_registry.get_initial_state();
  std::vector<int> feature = goose_heuristic->get_feature(initial_state);
  for (int i = 0; i < n_linear_models_; i++) {
    int h = goose_heuristic->compute_heuristic_from_feature(feature, i);
    open_lists[i].insert(h, initial_state.get_id());
  }

  if (!symmetry_) {
    /* no using WL symmetry */

    // just assume solvable problems
    while (true) {
      optional<SearchNode> node;
      StateID s_id = open_lists[q_cnt].remove_min();
      q_cnt = (q_cnt + 1) % n_linear_models_;
      State s = state_registry.lookup_state(s_id);
      node.emplace(search_space.get_node(s));
      node->close();
      statistics.inc_expanded();

      vector<OperatorID> applicable_ops;
      successor_generator.generate_applicable_ops(s, applicable_ops);

      for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];

        State succ_state = state_registry.get_successor_state(s, op);
        statistics.inc_generated();
        SearchNode succ_node = search_space.get_node(succ_state);

        if (succ_node.is_new()) {
          statistics.inc_evaluated_states(n_linear_models_);
          succ_node.open(*node, op, get_adjusted_cost(op));

          // must put here since we need to open the node before we can extract the plan
          if (check_goal_and_set_plan(succ_state)) {
            state = make_shared<State>(succ_state);
            return;
          }

          // generate WL features once to use for several linear models
          feature = goose_heuristic->get_feature(succ_state);
          for (int i = 0; i < n_linear_models_; i++) {
            int h = goose_heuristic->compute_heuristic_from_feature(feature, i);

            // log progress
            if (h < best_h[i]) {
              log << "New best heuristic value for h_" << i << ": " << h << std::endl;
              statistics.print_checkpoint_line(succ_node.get_g());
              best_h[i] = h;
            }

            // insert into priority queue
            open_lists[i].insert(h, succ_state.get_id());
          }
        }
      }
    }
  } else {
    /* using WL symmetry */

    std::seed_seq seed{0};
    std::mt19937 rng = std::mt19937(seed);

    int cnt_symmetries = 0;
    std::vector<int> hs;
    int h_adjustment;

    // just assume solvable problems
    while (true) {
      optional<SearchNode> node;
      StateID s_id = open_lists[q_cnt].remove_min();
      q_cnt = (q_cnt + 1) % n_linear_models_;
      State s = state_registry.lookup_state(s_id);
      node.emplace(search_space.get_node(s));
      node->close();
      statistics.inc_expanded();

      vector<OperatorID> applicable_ops;
      successor_generator.generate_applicable_ops(s, applicable_ops);
      // shuffle for symmetry breaking selection
      std::shuffle(applicable_ops.begin(), applicable_ops.end(), rng);

      std::map<std::vector<int>, std::vector<int>> seen_features;

      for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];

        State succ_state = state_registry.get_successor_state(s, op);
        statistics.inc_generated();
        SearchNode succ_node = search_space.get_node(succ_state);

        if (succ_node.is_new()) {
          statistics.inc_evaluated_states(n_linear_models_);
          succ_node.open(*node, op, get_adjusted_cost(op));

          // must put here since we need to open the node before we can extract the plan
          if (check_goal_and_set_plan(succ_state)) {
            state = make_shared<State>(succ_state);
            log << "WL symmetries detected: " << cnt_symmetries << std::endl;
            return;
          }

          // generate WL features once to use for several linear models
          feature = goose_heuristic->get_feature(succ_state);
          if (seen_features.count(feature)) {
            cnt_symmetries += 1;
            hs = seen_features[feature];
            h_adjustment = 1000000;
          } else {
            hs = std::vector<int>(n_linear_models_);
            for (int i = 0; i < n_linear_models_; i++) {
              hs[i] = goose_heuristic->compute_heuristic_from_feature(feature, i);
            }
            seen_features[feature] = hs;
            h_adjustment = 0;
          }

          for (int i = 0; i < n_linear_models_; i++) {
            int h = hs[i] + h_adjustment;  // 'delete' symmetries without losing completeness

            // log progress
            if (h < best_h[i]) {
              log << "New best heuristic value for h_" << i << ": " << h << std::endl;
              log << "g=" << succ_node.get_g() << ", "
                  << statistics.get_evaluated_states() << " evaluated, "
                  << statistics.get_expanded() << " expanded, "
                  << cnt_symmetries << " WL symmetries" 
                  << std::endl;
              best_h[i] = h;
            }

            // insert into priority queue
            open_lists[i].insert(h, succ_state.get_id());
          }
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
