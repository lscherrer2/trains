from torch_geometric.nn import GINEConv
from torch_geometric.data import Data
import torch.nn as nn


class GNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.ReLU(),
            nn.Linear(in_features=out_dim, out_features=out_dim),
        )
        self.conv = GINEConv(nn=self.mlp, edge_dim=edge_dim)

    def forward(self, data: Data) -> Data:
        return self.conv.forward(data.x, data.edge_index, data.edge_attr)
