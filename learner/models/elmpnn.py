import torch.nn

from .base_gnn import *
from torch_geometric.nn.conv import RGCNConv, FastRGCNConv  # (slow and/or mem inefficient)


class ELMPNNLayer(Module):
    def __init__(self, in_features: int, out_features: int, n_edge_labels: int, aggr: str):
        super(ELMPNNLayer, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(n_edge_labels):
            self.convs.append(LinearConv(in_features, out_features, aggr=aggr).jittable())
        self.root = Linear(in_features, out_features, bias=True)
        return

    def forward(self, x: Tensor, list_of_edge_index: List[Tensor]) -> Tensor:
        x_out = self.root(x)
        for i, conv in enumerate(self.convs):  # bottleneck
            x_out += conv(x, list_of_edge_index[i])
        return x_out


""" GNN with different weights for different edge labels """


class ELMPNN(BaseGNN):
    def __init__(self, params) -> None:
        super().__init__(params)
        if self.vn:
            raise NotImplementedError("vn not implemented for ELGNN")
        if self.share_layers:
            raise NotImplementedError("sharing layers not implemented for ELGNN")
        return

    def create_layer(self):
        return ELMPNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)

    def node_embedding(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
        """ overwrite (same semantics, different typing) for jit """
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, list_of_edge_index)
            x = F.relu(x)
        return x

    def graph_embedding(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
        """ overwrite (same semantics, different typing) for jit """
        x = self.node_embedding(x, list_of_edge_index, batch)
        x = self.pool(x, batch)
        return x

    def forward(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
        """ overwrite (same semantics, different typing) for jit """
        x = self.graph_embedding(x, list_of_edge_index, batch)
        x = self.mlp(x)
        return x


class ELMPNNPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        return

    def create_model(self, params):
        self.model = ELMPNN(params)

    def h(self, state: FrozenSet[Proposition]) -> float:
        x, edge_index = self.rep.get_state_enc(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None).item()
        h = round(h)
        return h

    def predict_action(self, state: FrozenSet[Proposition]):
        raise NotImplementedError


class ELMPNNRankerPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        return

    def create_model(self, params):
        self.model = ELMPNN(params)
        self.ranker = torch.nn.Linear(params["out_feat"], 1, bias=False)
        self.ranker_act = torch.nn.Identity()

    def forward(self, data):
        left = self.model.forward(data.x, data.edge_index, data.batch)
        right = self.model.forward(data.pair_x, data.pair_edge_index, data.batch)
        result = self.ranker_act(self.ranker(torch.sub(left, right))).squeeze(1)
        # print(left)
        # print(right)
        # print(data.)
        with torch.no_grad():
            if torch.sum(left - right) < 1e-3:
                print(f"Warning: Encodings are very close to each other: {left}, {right}")

            if torch.sum(result) < 1e-3:
                print(f"Warning: Classification is close to 0: {result}")
        return result

    def h(self, state: FrozenSet[Proposition]) -> float:
        x, edge_index = self.rep.get_state_enc(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None)
        # print(f"h: {h}")
        h = self.ranker(h).item()
        h = self.shift_heu(h)
        return h

    def h_batch(self, states: List[FrozenSet[Proposition]]) -> List[float]:
        data_list = []
        for state in states:
            x, edge_index = self.rep.get_state_enc(state)
            data_list.append(Data(x=x, edge_index=edge_index))
        loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
        data = next(iter(loader)).to(self.device)
        hs = self.model.forward(data.x, data.edge_index, data.batch)
        # print(f"model: {hs}")
        hs = self.ranker(hs)
        hs = hs.detach().cpu().numpy().reshape([-1, ])  # annoying error with jit
        # print(f"here: {hs}")
        hs = self.shift_heu(hs).tolist()
        # print(hs)
        return hs

    def shift_heu(self, h, scale=1e2, shift=10):
        shift_result = h + shift
        assert shift_result > 0, f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(np.exp(shift_result) * scale).astype("int32")
        assert result > 0, f"Invalid heuristic value: {result}"
        return result

    def predict_action(self, state: FrozenSet[Proposition]):
        raise NotImplementedError


class ELMPNNBatchedRankerPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        self.ranker = torch.nn.Linear(params["out_feat"], 1, bias=False)
        self.ranker_act = torch.nn.Tanh()
        return

    def create_model(self, params):
        self.model = ELMPNN(params)
        self.model.mlp = nn.Identity()

    def forward(self, data):

        with torch.no_grad():
            assert torch.sum(data.p_idx) - data.p_idx[0] * data.p_idx.shape[0] == 0
            # print(data.problem)
        encodes = self.ranker(self.model.forward(data.x, data.edge_index, data.batch))

        indices = torch.combinations(torch.arange(encodes.shape[0]), 2)
        combined_encodes = encodes[indices].permute([1, 0, 2])
        diff = combined_encodes[0, :] - combined_encodes[1, :]
        # print(polarity)
        result = self.ranker_act(diff).squeeze(1)
        with torch.no_grad():
            ys = data.y[indices].permute(1, 0)
            polarity_mask = ((ys[0, :] - ys[1, :]) > 0).long()
            polarity = torch.mul(torch.ones_like(polarity_mask) * 2, polarity_mask) - 1
            # polarity_mask = ((ys[0, :] - ys[1, :]) == 0).long()
            # polarity = polarity + polarity_mask
            polarity = polarity.float()
            # if torch.sum(torch.abs(diff)) < 1e-3:
            #     print(f"Warning: Encodings are very close to each other!: {torch.sum(torch.abs(diff)).detach().numpy()}")

            # if torch.sum(torch.abs(result)) < 1e-3:
            #     print(f"Warning: Classification is close to 0: {result}")
        return result, polarity

    def h(self, state: FrozenSet[Proposition]) -> float:
        x, edge_index = self.rep.get_state_enc(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None)
        # print(f"h: {h}")
        h = self.ranker(h).item()
        h = self.shift_heu(h)
        return h

    def h_batch(self, states: List[FrozenSet[Proposition]]) -> List[float]:
        data_list = []
        for state in states:
            x, edge_index = self.rep.get_state_enc(state)
            data_list.append(Data(x=x, edge_index=edge_index))
        loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
        data = next(iter(loader)).to(self.device)
        hs = self.model.forward(data.x, data.edge_index, data.batch)
        # print(f"model: {hs}")
        hs = self.ranker(hs)
        hs = hs.detach().cpu().numpy().reshape([-1, ])  # annoying error with jit
        # print(f"here: {hs}")
        hs = self.shift_heu(hs).tolist()
        # print(hs)
        return hs

    def shift_heu(self, h, scale=1e2, shift=10):
        shift_result = h + shift
        assert shift_result > 0, f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(np.exp(shift_result) * scale).astype("int32")
        assert result > 0, f"Invalid heuristic value: {result}"
        return result

    def predict_action(self, state: FrozenSet[Proposition]):
        raise NotImplementedError
