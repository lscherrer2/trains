from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch import Tensor
import torch
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


class Actor(nn.Module):
    def __init__(
        self, node_dim: int, edge_dim: int, switch_count: int, train_count: int
    ):
        super().__init__()
        self.g1 = GNN(in_dim=node_dim, out_dim=64, edge_dim=edge_dim)
        self.g2 = GNN(in_dim=64, out_dim=64, edge_dim=edge_dim)
        self.nn = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.nn_switches = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, switch_count),
            nn.Sigmoid(),
        )
        self.nn_trains = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, train_count),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        data.x = self.g1(data)
        data.x = self.g2(data)

        g = global_mean_pool(data.x, batch=data.batch)

        x = self.nn(g)
        switch_states = self.nn_switches(x)
        train_states = self.nn_trains(x)

        return switch_states.reshape((-1,)), train_states.reshape((-1,))
