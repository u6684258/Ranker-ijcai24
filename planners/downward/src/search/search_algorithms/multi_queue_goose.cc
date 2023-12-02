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

  State initial_state = state_registry.get_initial_state();
  std::vector<int> feature = goose_heuristic->get_feature(initial_state);

  for (int i = 0; i < n_linear_models_; i++) {
    int h = goose_heuristic->compute_heuristic_from_feature(feature, i);
    open_lists[i].insert(h, initial_state.get_id());
  }

  std::cout << "ok" << std::endl;

  // TODO from here
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
