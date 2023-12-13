
import logging
import torch
from dataset.graphs_hgn import create_input_and_target_hypergraphs_tuple
from representation.hypergraph_nets.delete_relaxation import DeleteRelaxationHypergraphView
from stripsHgn.Hgn import Hgn
from stripsHgn.HgnRanker import PlanRanker
from util.pyperplan_api import STRIPSProblem
from util.save_load import load_hgn_model


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
        problem: STRIPSProblem = STRIPSProblem(domain_file, problem_file)
        self.hypergraph = (
            DeleteRelaxationHypergraphView(problem)
        )

        # Load STRIPS-HGN model and setup evaluation mode
        if model == "hgn":
            self.model: Hgn = load_hgn_model(checkpoint)
            self.type = "hgn"
        if model == "ranker":
            # print(checkpoint)
            self.model: PlanRanker = load_hgn_model(checkpoint)
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
        elif self.type == "ranker":
            output_h, _ = self.model(input_h_tuple)
            assert output_h.item() > 0
            return output_h


if __name__ == "__main__":
    domain = "/home/ming/PycharmProjects/DirectRankerNew/Data/exp_set/blocks/domain.pddl"
    problem = "/home/ming/PycharmProjects/DirectRankerNew/Data/exp_set/blocks/train/blocks3-task01.pddl"
    checkpoint = "/home/ming/PycharmProjects/DirectRankerNew/experiments/2023-07-10T22:31:31.501143/train-hgn-ranker-2023-07-13T01:32:06.002717/model-mix-best.ckpt"
    evaluator = HGNEvaluator(domain, problem, checkpoint, model="ranker")
    print(evaluator.evaluate_state({}, 10))
