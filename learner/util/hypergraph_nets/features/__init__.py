from abc import ABC, abstractmethod
from typing import List

from hgn.hypergraph_nets import Node, Hyperedge, Number

class AbstractFeatureMapper(ABC):
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def input_size(cls) -> int:
        """ Size of the feature vector returned by this feature mapper """
        raise NotImplementedError


class GlobalFeatureMapper(AbstractFeatureMapper, ABC):
    @abstractmethod
    def hypergraph_view_to_feature(
        self, hypergraph_view
    ) -> List[Number]:
        raise NotImplementedError

    def __call__(self, hypergraph_view) -> List[Number]:
        return self.hypergraph_view_to_feature(hypergraph_view)


class NodeFeatureMapper(AbstractFeatureMapper, ABC):
    @abstractmethod
    def node_to_feature(self, node: Node) -> List[Number]:
        raise NotImplementedError

    def __call__(self, node: Node) -> List[Number]:
        return self.node_to_feature(node)


class HyperedgeFeatureMapper(AbstractFeatureMapper, ABC):
    @abstractmethod
    def hyperedge_to_feature(self, hyperedge: Hyperedge) -> List[Number]:
        raise NotImplementedError

    def __call__(self, hyperedge: Hyperedge) -> List[Number]:
        return self.hyperedge_to_feature(hyperedge)
