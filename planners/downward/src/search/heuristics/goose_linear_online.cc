#include "goose_linear_online.h"

#include <random>

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"

namespace goose_linear_online {

GooseLinearOnline::GooseLinearOnline(const plugins::Options &opts)
    : goose_linear::GooseLinear(opts) {
  train();
}

FullState GooseLinearOnline::assign_random_state(const PartialState &state) {
  FullState ret;

  for (int var = 0; var < n_variables; var++) {  // TODO(DZC) can optimise if -1 vars are static?
    int val = state[var];
    if (val == -1) {
      std::uniform_int_distribution<std::mt19937::result_type> dist(0, vars[var].get_domain_size() -
                                                                           1);
      val = dist(rng);
    }
    ret.push_back(FactPair(var = var, val = val));
  }

  return ret;
}

void GooseLinearOnline::train() {
  n_variables = task->get_num_variables();
  vars = task_proxy.get_variables();
  rng = std::mt19937(dev());

  // initial state in backwards search is the goal condition
  std::map<VariableProxy, int> var_to_ind;
  for (int i = 0; i < n_variables; i++) {
    var_to_ind[vars[i]] = i;
  }

  PartialState goal_condition(n_variables, -1);
  for (FactProxy goal : task_proxy.get_goals()) {
    goal_condition[var_to_ind[goal.get_variable()]] = goal.get_value();
  }

  std::cout << goal_condition << std::endl;

  FullState full_state = assign_random_state(goal_condition);
  int y = 0;
  SearchNodeStats stats = compute_heuristic_vector_state(full_state);

  BackwardsSearchNode node(goal_condition, y, stats);

  std::set<PartialState> seen;
  std::queue<BackwardsSearchNode> q;

  q.push(node);
  while (!q.empty() && seen.size() < 10000) {
    BackwardsSearchNode node = q.front();
    q.pop();
    PartialState partial_state = node.state;
    int y = node.y;

    for (const auto &op : task_proxy.get_operators()) {
      std::cout << op.get_name() << std::endl;
    }
  }

  exit(-1);

  // BackwardsSearchNode init_node(goal_condition, 0, 0, 0);

  // for (FactProxy goal : task.get_goals()) {
  //     if (state[goal.get_variable()] != goal)
  //         return false;
  // }
  // return true;
}

SearchNodeStats GooseLinearOnline::compute_heuristic_vector_state(const FullState &state) {
  std::vector<long> cur_seen_colours = cnt_seen_colours;
  std::vector<long> cur_unseen_colours = cnt_unseen_colours;
  std::vector<double> ratio(iterations_);
  CGraph graph = fact_pairs_to_graph(state);
  std::vector<int> feature = wl_feature(graph);
  int h = predict(feature);
  for (size_t i = 0; i < iterations_; i++) {
    cur_seen_colours[i] -= cnt_seen_colours[i];
    cur_unseen_colours[i] -= cnt_unseen_colours[i];
    ratio[i] = cur_seen_colours[i] / (cur_seen_colours[i] + cur_unseen_colours[i]);
  }
  return SearchNodeStats(h = h, ratio = ratio);
}

class GooseLinearOnlineFeature : public plugins::TypedFeature<Evaluator, GooseLinearOnline> {
 public:
  GooseLinearOnlineFeature() : TypedFeature("linear_model_online") {
    document_title("GOOSE optimised WL feature linear model heuristic with online training");
    document_synopsis("TODO");

    // https://github.com/aibasel/downward/pull/170 for string options
    add_option<std::string>("model_file", "path to trained python model", "default_value");
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

static plugins::FeaturePlugin<GooseLinearOnlineFeature> _plugin;

}  // namespace goose_linear_online
