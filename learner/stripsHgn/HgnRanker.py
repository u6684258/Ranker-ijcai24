import json
import logging
from numbers import Number
from typing import List, Any, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stripsHgn.direct_ranker import DirectRanker, FFN, FFNGraph
from representation.hypergraph_nets.hypergraphs import HypergraphsTuple
from representation.hypergraph_nets.models import EncodeProcessDecode
from util import Namespace

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_LATENT_SIZE = 64


def _get_debug_dict(hparams):
    return {
        "latent_size": hparams["latent_size"],
        "batch_size": hparams["batch_size"],
        "num_steps": hparams["num_steps"],
        "receiver_k": hparams["receiver_k"],
        "sender_k": hparams["sender_k"],
        "hidden_size": hparams["hidden_size"],
    }


class PlanRanker(torch.nn.Module):
    """

    - base_network: the encoder of the ranker, expecting a STRIPSHGN
    - hyperparameters:
        - latent_size: output size of the base network
        - antisymmetric_activation: final activation layer. Must be an antisymmetric function
        - loss_function: loss function
        - batch_size: batch_size
        - learning_rate: learning_rate
        - weight_decay: weight_decay
        - num_steps: number of message passing steps in STRIPSHGN
        hyperparameters for STRIPSHGN:
        - receiver_k
        - sender_k
        - hidden_size
        - global_feature_mapper_cls
        - hyperedge_feature_mapper_cls
        - node_feature_mapper_cls
    """

    def __init__(self, params):
        super().__init__()
        self.hparams = {}
        for key in params.keys():
            self.hparams[key] = params[key]
        self._prediction_mode = False
        self.show_countdown = 0

        if self.hparams["model"] == "HGNNRANK":
            # Setup HGN
            self.base_network = EncodeProcessDecode(
                receiver_k=params.receiver_k,
                sender_k=params.sender_k,
                hidden_size=params.hidden_size,
                global_input_size=params.global_feature_mapper_cls.input_size(),
                edge_input_size=params.hyperedge_feature_mapper_cls.input_size(),
                node_input_size=params.node_feature_mapper_cls.input_size(),
                global_output_size=params.latent_size,
                last_relu=False
            )

        elif self.hparams["model"] == "ffn":
            self.base_network = FFNGraph(
                input_dims=[params.num_actns,
                            params.hyperedge_feature_mapper_cls.input_size(),
                            params.num_props,
                            params.node_feature_mapper_cls.input_size(),
                            ],
                output_dim=params.latent_size)

        self.ranker = DirectRanker(
            latent_size=self.hparams["latent_size"],
            antisymmetric_activation=torch.tanh,
        )

        # Log hyperparameters
        print(
            "Direct Ranker hparams:\n"
            f"{json.dumps(_get_debug_dict(params), indent=2)}"
        )

        self.base_network.to(self.hparams["device"])
        self.ranker.to(self.hparams["device"])
        self.to(self.hparams["device"])

    def setup_prediction_mode(self):
        self._prediction_mode = True
        self.ranker.setup_prediction_mode()

    def get_num_parameters(self) -> int:
        """ Count number of weight parameters """
        # https://stackoverflow.com/a/62764464/13531424
        # e.g. to deal with case of sharing layers
        params = sum(dict((p.data_ptr(), p.numel()) for p in self.parameters() if p.requires_grad).values())
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return params

    def forward(self, hypergraphs: HypergraphsTuple):
        """
        Run the forward pass through the network

        Parameters
        ----------
        hypergraph_left, hypergraph_right: HypergraphsTuple, one or more hypergraphs

        Returns
        -------
        Output rank order prediction
        """
        # print(hypergraph_left)
        # self.show_countdown += 1
        left = self.base_network(
            hypergraph=hypergraphs,
            steps=self.hparams["num_steps"],
            pred_mode=self._prediction_mode,
        )
        if not self._prediction_mode:
            # if self.show_countdown % 1000 == 0:
            #     print(f"Encodes: {left[-1].globals} ")
            results = []
            polaritys = []
            for i in range(len(left)):
                result, polarity = self.ranker(hypergraphs, left[i].globals)
                results.append(result)
                polaritys.append(polarity)
            return results, polaritys

        else:
            pred, polarity = self.ranker(hypergraphs, left[-1].globals)
            # if self.show_countdown % 100 == 0:
            #     _log.debug(f"hgn output: {left[-1].globals.detach().numpy()}, ranker output: {pred.detach().numpy()}")

            return pred, polarity

    def training_step(self, batch, batch_idx):

        preds, targets = self.forward(batch)
        # print(f"targets: {np.around(targets.detach().numpy(), 5)}, pred: {np.around(preds.detach().numpy(), 5)}")
        loss = self._calc_loss(preds, targets)
        self.log("loss", loss.mean())
        self.log("prediction", np.around(preds[-1].detach().numpy(), 4).item())
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([step["loss"] for step in outputs]).mean()
        tensorboard_logs = {"train_loss": avg_loss}
        self.show_countdown = 0
        self.log("avg_train_loss", avg_loss)
        self.log("log", tensorboard_logs)
        # return {"avg_train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # Run model and calculate loss
        preds, targets = self(batch)
        loss = self._calc_loss(preds, targets)
        # print(f"{preds}, {loss}")
        self.log("val_loss", loss.mean())
        return loss

    def validation_epoch_end(self, outputs):
        self.show_countdown = 0
        avg_loss = torch.stack(outputs).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.log("avg_val_loss", avg_loss)
        self.log("log", tensorboard_logs)
        # return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    #     return self(self._prepare_batch(batch))

