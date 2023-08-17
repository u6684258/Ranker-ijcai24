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
    self.ranker_act = torch.nn.Tanh()

  def forward(self, data):
    left = self.model.forward(data.x, data.edge_index, data.batch)
    right = self.model.forward(data.pair_x, data.pair_edge_index, data.batch)
    return self.ranker_act(self.ranker(torch.sub(left, right))).squeeze(1)

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
    print(f"model: {hs}")
    hs = self.ranker(hs)
    hs = hs.detach().cpu().numpy().reshape([-1,])  # annoying error with jit
    print(f"here: {hs}")
    hs = self.shift_heu(hs).tolist()
    print(hs)
    return hs

  def shift_heu(self, h, scale=1e2, shift=10):
    return np.round(np.exp(h+shift)*scale).astype("int32")

  def predict_action(self, state: FrozenSet[Proposition]):
    raise NotImplementedError
