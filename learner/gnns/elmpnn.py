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
        for i, conv in enumerate(self.convs):  # bottleneck; difficult to parallelise efficiently
            x_out += conv(x, list_of_edge_index[i])
        return x_out


""" GNN with different weights for different edge labels """


class ELMPNN(BaseGNN):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.is_ranker = False
        if self.vn:
            raise NotImplementedError("vn not implemented for ELGNN")
        if self.share_layers:
            raise NotImplementedError("sharing layers not implemented for ELGNN")
        return

    def create_layer(self):
        return ELMPNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)

    def node_embedding(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
        """ overwrite typing (same semantics, different typing) for jit """
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, list_of_edge_index)
            if self.is_ranker:
                x = F.leaky_relu(x)
            else:
                x = F.relu(x)
        return x

    def graph_embedding(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
        """ overwrite typing (same semantics, different typing) for jit """
        x = self.node_embedding(x, list_of_edge_index, batch)
        x = self.pool(x, batch)
        return x

    def forward(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
        """ overwrite typing (same semantics, different typing) for jit """
        x = self.graph_embedding(x, list_of_edge_index, batch)
        x = self.mlp(x)
        x = self.last_layer(x)
        if x.size()[1] == 1:
            x = x.squeeze(1)
        return x


class ELMPNNPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        return

    def create_model(self, params):
        self.model = ELMPNN(params)

    def h(self, state: State) -> float:
        x, edge_index = self.rep.state_to_tensor(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None).item()
        h = round(h)
        return h

    def predict_action(self, state: State):
        raise NotImplementedError


class ELMPNNRankerPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        return

    def create_model(self, params):
        self.model = ELMPNN(params)
        self.model.last_layer = nn.Identity()
        self.ranker = torch.nn.Linear(params["out_feat"], 1, bias=False)
        self.ranker_act = torch.tanh

    def forward(self, data):
        encodes = self.model.forward(data.x, data.edge_index, data.batch)
        indices = torch.combinations(torch.arange(encodes.shape[0]), 2)
        # indices = torch.tensor([(i, j) for i, j in zip(torch.arange(encodes.shape[0]),
        #                                   torch.arange(encodes.shape[0])[1:])]).reshape([-1, 2])
        combined_encodes = encodes[indices].permute([1, 0, 2])
        diff = combined_encodes[1, :] - combined_encodes[0, :]
        # print(polarity)
        result = self.ranker_act(self.ranker(diff)).squeeze(1)
        # print(left)
        # print(right)
        # print(data.)
        with torch.no_grad():
            ys = data.y[indices].permute(1, 0)
            polarity_mask = ((ys[1, :] - ys[0, :]) > 0).long()
            polarity = 2 * polarity_mask - 1
            polarity = polarity.float()
            if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                print(f"Warning: Encodings are very close to each other: {diff}")

            if torch.sum(torch.abs(result)) / diff.shape[0] < 1e-3:
                print(f"Warning: Classification is close to 0: {result}")
        return result, polarity

    def h(self, state: State) -> float:
        x, edge_index = self.rep.state_to_tensor(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None)
        h = self.ranker(h).detach().cpu().numpy().reshape([-1, ])
        # print(f"h: {h}")
        h = self.shift_heu(h)
        return h

    def h_batch(self, states: State) -> List[float]:
        data_list = []
        for state in states:
            x, edge_index = self.rep.state_to_tensor(state)
            data_list.append(Data(x=x, edge_index=edge_index))
        loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
        hs_all = []
        for data in loader:
            data = data.to(self.device)
            hs = self.model.forward(data.x, data.edge_index, data.batch)
            hs = self.ranker(hs)
            hs = hs.detach().cpu().numpy()  # annoying error with jit
            hs_all.append(hs)
        hs_all = np.concatenate(hs_all).astype(float).reshape([-1, ])
        # print(f"here: {hs}")
        hs = self.shift_heu(hs_all).tolist()
        # print(hs)
        return hs

    def shift_heu(self, h, scale=1e3, shift=1e4):
        result = h + shift
        # print(f"result: {result}")
        assert (2147483647 > result).all() and (
                    result > 0).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all() and (result > 0).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result

    def predict_action(self, state: FrozenSet[Proposition]):
        raise NotImplementedError


class ELMPNNBatchedRankerPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        return

    def create_model(self, params):
        self.model = ELMPNN(params)
        self.model.mlp = Sequential(
            Linear(self.model.nhid, self.model.nhid),
            torch.nn.LeakyReLU(),
        )
        self.model.last_layer = torch.nn.Linear(params["out_feat"], params["out_feat"])
        self.model.ranker = torch.nn.Linear(params["out_feat"], 1, bias=False)
        self.model.ranker_act = torch.nn.Sigmoid()
        self.model.is_ranker = True
    def forward(self, data):

        with torch.no_grad():
            assert torch.sum(data.p_idx) - data.p_idx[0] * data.p_idx.shape[0] == 0
            # print(data.problem)
        encodes = self.model.forward(data.x, data.edge_index, data.batch)

        indices = torch.combinations(torch.arange(encodes.shape[0]), 2)
        indices = torch.tensor([(i, j) for i, j in zip(torch.arange(encodes.shape[0]),
                                                       torch.arange(encodes.shape[0])[1:])]).reshape([-1, 2])

        combined_encodes = encodes[indices].permute([1, 0, 2])
        diff = combined_encodes[0, :] - combined_encodes[1, :]
        # print(polarity)
        result = self.model.ranker_act(self.model.ranker(diff)).squeeze(1)
        with torch.no_grad():
            ys = data.y[indices].permute(1, 0)
            polarity_mask = ((ys[0, :] - ys[1, :]) > 0).long()
            polarity = 2 * polarity_mask - 1
            polarity = polarity.float()
            if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                print(f"Warning: Encodings are very close to each other: {diff}")

            if torch.sum(torch.abs(result)) / diff.shape[0] < 1e-3:
                print(f"Warning: Classification is close to 0: {result}")
        return result, polarity

    def h(self, state: State) -> float:
        x, edge_index = self.rep.state_to_tensor(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None)
        h = self.model.ranker(h).detach().cpu().numpy().reshape([-1, ])
        # print(f"h: {h}")
        h = self.shift_heu(h)
        return h

    def h_batch(self, states: State) -> List[float]:
        data_list = []
        for state in states:
            x, edge_index = self.rep.state_to_tensor(state)
            data_list.append(Data(x=x, edge_index=edge_index))
        loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
        hs_all = []
        for data in loader:
            data = data.to(self.device)
            hs1 = self.model.forward(data.x, data.edge_index, data.batch)
            hs2 = self.model.ranker(hs1)
            hs = hs2.detach().cpu().numpy()  # annoying error with jit
            hs_all.append((hs))
        hs_all = np.concatenate(hs_all).astype(float).reshape([-1, ])
        # print(f"here: {hs}")
        hs = self.shift_heu(hs_all).tolist()
        # print(hs)
        return hs

    def shift_heu(self, h, scale=1e3, shift=1e5):
        result = h + shift
        # print(f"result: {result}")
        assert (2147483647 > result).all() and (
                    result > 0).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all() and (result > 0).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result

    def predict_action(self, state: State):
        raise NotImplementedError


class ELMPNNBatchedCoordRankerPredictor(ELMPNNBatchedRankerPredictor):

    def forward(self, data):
        with torch.no_grad():
            assert torch.sum(data.p_idx) - data.p_idx[0] * data.p_idx.shape[0] == 0
            # print(data.problem)
        encodes = self.model.forward(data.x, data.edge_index, data.batch)

        encodes_xy = torch.concatenate([encodes, data.coord_x.reshape([-1, 1]), data.coord_y.reshape([-1,1])], dim=1)
        unique = torch.unique(data.coord_x)
        split_by_x = [encodes_xy[data.coord_x == i] for i in unique]
        diff = []
        for s in split_by_x:
            sort_s = s[s[:, -1].sort()[1]]
            diff.append(sort_s[1:, :-2] - sort_s[0, :-2])
        diff = torch.concatenate(diff,  dim=0)
        assert diff.size(0) + unique.size(0) == encodes.size(0)

        result = self.model.ranker_act(self.model.ranker(diff)).squeeze(1)
        with torch.no_grad():
            polarity = torch.ones(diff.size(0))
            if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                print(f"Warning: Encodings are very close to each other: {diff}")

            if torch.sum(torch.abs(result)) / diff.shape[0] < 1e-3:
                print(f"Warning: Classification is close to 0: {result}")
        return result, polarity


class ELMPNNBatchedProbPredictor(BasePredictor):
    def __init__(self, params, jit=False) -> None:
        super().__init__(params, jit)
        self.ranker = torch.nn.Linear(params["out_feat"], 1, bias=False)
        self.ranker_act = torch.nn.Sigmoid()
        return

    def create_model(self, params):
        self.model = ELMPNN(params)
        self.model.mlp = nn.Identity()

    def forward(self, data):

        with torch.no_grad():
            assert torch.sum(data.p_idx) - data.p_idx[0] * data.p_idx.shape[0] == 0
            # print(data.problem)
        encodes = self.model.forward(data.x, data.edge_index, data.batch)

        # indices = torch.combinations(torch.arange(encodes.shape[0]), 2)
        indices = torch.tensor([(i, j) for i, j in zip(torch.arange(encodes.shape[0]),
                                                       torch.arange(encodes.shape[0])[1:])]).reshape([-1, 2])

        combined_encodes = encodes[indices].permute([1, 0, 2])
        diff = combined_encodes[0, :] - combined_encodes[1, :]
        # print(polarity)
        result = self.ranker_act(self.ranker(diff)).squeeze(1)
        with torch.no_grad():
            ys = data.y[indices].permute(1, 0)
            polarity = ((ys[0, :] - ys[1, :]) > 0).float()
            if torch.sum(torch.abs(diff)) / diff.shape[0] < 1e-3:
                print(f"Warning: Encodings are very close to each other: {diff}")

            if torch.sum(torch.abs(result)) / diff.shape[0] < 1e-3:
                print(f"Warning: Classification is close to 0: {result}")
        return result, polarity

    def h(self, state: State) -> float:
        x, edge_index = self.rep.state_to_tensor(state)
        x = x.to(self.device)
        for i in range(len(edge_index)):
            edge_index[i] = edge_index[i].to(self.device)
        h = self.model.forward(x, edge_index, None)
        h = self.ranker(h).detach().cpu().numpy().reshape([-1, ])
        # print(f"h: {h}")
        h = self.shift_heu(h)
        return h

    def h_batch(self, states: State) -> List[float]:
        data_list = []
        for state in states:
            x, edge_index = self.rep.state_to_tensor(state)
            data_list.append(Data(x=x, edge_index=edge_index))
        loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
        hs_all = []
        for data in loader:
            data = data.to(self.device)
            hs = self.model.forward(data.x, data.edge_index, data.batch)
            hs = self.ranker(hs)
            hs = hs.detach().cpu().numpy()  # annoying error with jit
            hs_all.append(hs)
        hs_all = np.concatenate(hs_all).astype(float).reshape([-1, ])
        # print(f"here: {hs}")
        hs = self.shift_heu(hs_all).tolist()
        # print(hs)
        return hs

    def shift_heu(self, h, scale=1e3, shift=1e4):
        result = h + shift
        # print(f"result: {result}")
        assert (2147483647 > result).all() and (
                    result > 0).all(), f"shift {shift} is not large enough to make {h} a positive heuristic values"
        result = np.round(result * scale).astype("int32")
        assert (2147483647 > result).all() and (result > 0).all(), f"Invalid heuristic value: {result}; Origin: {h}"
        return result

    def predict_action(self, state: State):
        raise NotImplementedError
