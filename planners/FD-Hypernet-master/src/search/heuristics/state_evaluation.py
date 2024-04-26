
import logging
import os

import torch

from dataset.dataset_hgn import create_input_and_target_hypergraphs_tuple
from util.hypergraph_nets.delete_relaxation import DeleteRelaxationHypergraphView
from models.model_hgn import HgnModel
from models.rank_model_hgn import HgnRankModel
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
        if model == "hgn" or  model == "hgn-loss":
            self.model: HgnModel = load_hgn_model(checkpoint)
            self.type = "hgn"
        if model == "hgn-rank":
            # print(checkpoint)
            self.model: HgnRankModel = load_hgn_model(checkpoint)
            self.type = "hgn-rank"
        self.model.setup_prediction_mode()

    def evaluate_state(self, state, num_steps: int = 10):
        # Call STRIPSHGN directly with a HypergraphsTuple
        input_h_tuple = create_input_and_target_hypergraphs_tuple(state, self.hypergraph,
                                                                  self.model.hparams["receiver_k"],
                                                                  self.model.hparams["sender_k"], [0,0], 0)
        if self.type == "hgn":
            output_h, _ = self.model(input_h_tuple)
            assert output_h.item() >= 0
            return output_h.item()
        elif self.type == "hgn-rank":
            output_h, _ = self.model(input_h_tuple)
            assert output_h.item() >= 0
            # print(output_h)
            return output_h.item()


if __name__ == "__main__":
    domain = "/home/ming/PycharmProjects/goose/benchmarks/ipc2023-learning-benchmarks/blocksworld/domain.pddl"
    problem = "/home/ming/PycharmProjects/goose/benchmarks/ipc2023-learning-benchmarks/blocksworld/testing/easy/p01.pddl"
    checkpoint = "/home/ming/PycharmProjects/goose/logs/hgn_gnn_models/blocksworld_llg_L4_H64_mean_r0_hgn-rank.dt"
    evaluator = HGNEvaluator(domain, problem, checkpoint, model="hgn-rank")
    print(evaluator.evaluate_state({'(on b3 b5)', '(on b2 b1)', '(clear b2)', '(arm-empty )', '(on-table b1)', '(on-table b4)', '(clear b3)', '(on b5 b4)'}
, 10))
