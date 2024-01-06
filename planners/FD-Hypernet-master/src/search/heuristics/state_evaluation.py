
import logging
import os

import torch

from hgn.dataset_hgn import create_input_and_target_hypergraphs_tuple
from hgn.hypergraph_nets.delete_relaxation import DeleteRelaxationHypergraphView
from hgn.model_hgn import HgnModel
from hgn.rank_model_hgn import HgnRankModel
from util.mdpsim_api import STRIPSProblem
from util.save_load import load_hgn_model, load_gnn_model


class HGNEvaluator:
    def __init__(
            self, domain_file, problem_file, checkpoint, multithreading=False, sas=False, model="hgn"
    ):
        """ Runs STRIPS-HGN for initial state of problem """
        # Fix number of threads used for pytorch
        if not multithreading:
            torch.set_num_threads(1)
        # Set logging level to info
        logging.basicConfig(level=logging.INFO)
        # Generate the STRIPSProblem and get the DeleteRelaxationHypergraphView
        problem = STRIPSProblem(domain_file, [problem_file])
        problem.change_problem(problem_file)
        self.hypergraph = (
            DeleteRelaxationHypergraphView(problem)
        )

        # Load STRIPS-HGN model and setup evaluation mode
        if model == "hgn":
            self.model: HgnModel = load_hgn_model(checkpoint)
            self.type = "hgn"
        if model == "hgn-rank":
            # print(checkpoint)
            self.model: HgnRankModel = load_hgn_model(checkpoint)
            self.type = "ranker"
        self.model.setup_prediction_mode()

    def evaluate_state(self, state, num_steps: int = 10):
        # Call STRIPSHGN directly with a HypergraphsTuple
        input_h_tuple = create_input_and_target_hypergraphs_tuple(state, self.hypergraph,
                                                                  self.model.hparams["receiver_k"],
                                                                  self.model.hparams["sender_k"], [0,0], 0)
        if self.type == "hgn":
            output_h, _ = self.model(input_h_tuple)
            assert output_h.item() > 0
            return output_h
        elif self.type == "hgn-rank":
            output_h, _ = self.model(input_h_tuple)
            assert output_h.item() > 0
            return output_h


if __name__ == "__main__":
    domain = "/home/ming/PycharmProjects/goose/benchmarks/ipc2023-learning-benchmarks/blocksworld/domain.pddl"
    problem = "/home/ming/PycharmProjects/goose/benchmarks/ipc2023-learning-benchmarks/blocksworld/testing/easy/p01.pddl"
    checkpoint = "/home/ming/PycharmProjects/goose/logs/hgn_gnn_models/blocksworld_llg_L4_H64_mean_r0_hgn.dt"
    evaluator = HGNEvaluator(domain, problem, checkpoint, model="hgn")
    print(evaluator.evaluate_state({}, 10))
