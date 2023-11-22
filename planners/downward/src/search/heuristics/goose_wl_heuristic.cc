#include "goose_wl_heuristic.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"

using std::string;

namespace goose_wl {

WLGooseHeuristic::WLGooseHeuristic(const plugins::Options &opts)
    : goose_heuristic::GooseHeuristic(opts) {}

CGraph WLGooseHeuristic::state_to_graph(const State &state) {
  std::vector<std::vector<std::pair<int, int>>> edges = graph_.get_edges();
  std::vector<int> colours = graph_.get_colours();
  int cur_node_fact;
  int new_idx = graph_.n_nodes();

  std::pair<std::string, std::vector<std::string>> pred_args;
  std::string pred, node_name;
  std::vector<std::string> args;
  for (const FactProxy &fact : convert_ancestor_state(state)) {
    pred_args = fact_to_lifted_input[fact.get_pair()];
    pred = pred_args.first;
    args = pred_args.second;
    if (pred.size() == 0) {
      continue;
    }

    node_name = pred;
    for (const std::string &arg : args) {
      node_name += ',' + arg;
    }

    if (graph_.is_pos_goal_node(node_name)) {
      colours[graph_.get_node_index(node_name)] = graph_.TRUE_POS_GOAL_;
      continue;
    }
    if (graph_.is_neg_goal_node(node_name)) {
      colours[graph_.get_node_index(node_name)] = graph_.TRUE_NEG_GOAL_;
      continue;
    }

    // add new node
    cur_node_fact = new_idx;
    new_idx++;
    colours.push_back(0);  // TRUE_FACT
    std::vector<std::pair<int, int>> new_edges_fact;
    edges.push_back(new_edges_fact);

    // connect fact to predicate
    int pred_node = graph_.get_node_index(pred);
    edges[cur_node_fact].push_back(std::make_pair(pred_node, graph_.GROUND_EDGE_LABEL_));
    edges[pred_node].push_back(std::make_pair(cur_node_fact, graph_.GROUND_EDGE_LABEL_));

    for (size_t k = 0; k < args.size(); k++) {
      // connect fact to object
      int object_node = graph_.get_node_index(args[k]);
      edges[object_node].push_back(std::make_pair(cur_node_fact, k));
      edges[cur_node_fact].push_back(std::make_pair(object_node, k));
    }
  }

  return {edges, colours};
}

std::vector<int> WLGooseHeuristic::wl_feature(const CGraph &graph) {
  if (wl_algorithm_ == "1wl") {
    return wl1_feature(graph);
  } else {
    std::cout << "error: encountered invalid WL algorithm " << wl_algorithm_ << std::endl;
    exit(-1);
  }
  return std::vector<int>();
}

std::vector<int> WLGooseHeuristic::wl1_feature(const CGraph &graph) {
  // feature to return is a histogram of colours seen during training
  std::vector<int> feature(feature_size_, 0);

  const size_t n_nodes = graph.n_nodes();

  // role of colours_0 and colours_1 is switched every iteration for storing old and new colours
  std::vector<int> colours_0(n_nodes);
  std::vector<int> colours_1(n_nodes);
  std::vector<std::vector<std::pair<int, int>>> edges = graph.get_edges();

  // determine size of neighbour colours from the start
  std::vector<std::vector<std::pair<int, int>>> neighbours = edges;

  int col = -1;
  std::string new_colour;

  // collect initial colours
  for (size_t u = 0; u < n_nodes; u++) {
    // initial colours always in hash and hash value always within size
    col = hash_[std::to_string(graph.colour(u))];
    feature[col]++;
    colours_0[u] = col;
  }

  // main WL algorithm loop
  for (size_t itr = 0; itr < iterations_; itr++) {
    // instead of assigning colours_0 = colours_1 at the end of every loop
    // we just switch the roles of colours_0 and colours_1 every loop
    if (itr % 2 == 0) {
      for (size_t u = 0; u < n_nodes; u++) {
        // we ignore colours we have not seen during training
        if (colours_0[u] == -1) {
          goto end_of_loop0;
        }

        // collect colours from neighbours and sort
        for (size_t i = 0; i < edges[u].size(); i++) {
          col = colours_0[edges[u][i].first];
          if (col == -1) {
            goto end_of_loop0;
          }
          neighbours[u][i] = std::make_pair(col, edges[u][i].second);
        }
        sort(neighbours[u].begin(), neighbours[u].end());

        // add current colour and sorted neighbours into sorted colour key
        new_colour = std::to_string(colours_0[u]);
        for (const auto &ne_pair : neighbours[u]) {
          new_colour += "," + std::to_string(ne_pair.first) + "," + std::to_string(ne_pair.second);
        }

        // hash seen colours
        if (hash_.count(new_colour)) {
          col = hash_[new_colour];
          feature[col]++;
        } else {
          col = -1;
        }
      end_of_loop0:
        colours_1[u] = col;
      }
    } else {
      for (size_t u = 0; u < n_nodes; u++) {
        // we ignore colours we have not seen during training
        if (colours_1[u] == -1) {
          goto end_of_loop1;
        }

        // collect colours from neighbours and sort
        for (size_t i = 0; i < edges[u].size(); i++) {
          col = colours_1[edges[u][i].first];
          if (col == -1) {
            goto end_of_loop1;
          }
          neighbours[u][i] = std::make_pair(col, edges[u][i].second);
        }
        sort(neighbours[u].begin(), neighbours[u].end());

        // add current colour and sorted neighbours into sorted colour key
        new_colour = std::to_string(colours_1[u]);
        for (const auto &ne_pair : neighbours[u]) {
          new_colour += "," + std::to_string(ne_pair.first) + "," + std::to_string(ne_pair.second);
        }

        // hash seen colours
        if (hash_.count(new_colour)) {
          col = hash_[new_colour];
          feature[col]++;
        } else {
          col = -1;
        }
      end_of_loop1:
        colours_0[u] = col;
      }
    }
  }

  return feature;
}

}  // namespace goose_wl
