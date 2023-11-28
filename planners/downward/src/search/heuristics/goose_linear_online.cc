#include "goose_linear_online.h"

#include <random>
#include <unordered_set>

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

inline bool regressable(const PartialState &state, const OperatorProxy &op) {
  // https://fai.cs.uni-saarland.de/teaching/winter18-19/planning-material/planning06-progression-and-regression-post-handout.pdf
  // slide 16/32
  bool non_empty = false;

  std::unordered_set<int> effect_vars;

  FactPair fact_pair;
  for (const EffectProxy &eff : op.get_effects()) {
    fact_pair = eff.get_fact().get_pair();  // assume no conditional effects
    int var = fact_pair.var;
    int g_v = state[var];
    int eff_v = fact_pair.value;
    effect_vars.insert(var);

    // (i) effect and partial state non empty
    if (g_v == eff_v) {
      non_empty = true;
      break;
    }

    // (ii) effect leads into the partial state
    if (g_v != -1 && g_v != eff_v) {
      return false;
    }
  }

  if (!non_empty) {
    return false;
  }

  for (const FactProxy& fact : op.get_preconditions()) {
    fact_pair = fact.get_pair();  // assume no conditional effects
    int var = fact_pair.var;
    int g_v = state[var];
    int pre_v = fact_pair.value;

    // (iii) unchanged precondition still the same in the partial state
    if (!effect_vars.count(var) && g_v != -1 && pre_v != g_v) {
      return false;
    }
  }

  return true;
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
      if (regressable(partial_state, op))
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
