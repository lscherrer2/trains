from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from trains.rl.util import GNN
from torch import Tensor
import torch.nn as nn
import torch


class Actor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        switch_count: int,
        train_count: int,
    ):
        super().__init__()
        self.g1 = GNN(in_dim=node_dim, out_dim=64, edge_dim=edge_dim)
        self.g2 = GNN(in_dim=64, out_dim=64, edge_dim=edge_dim)
        self.nn = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
        )

        self.nn_switches = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(switch_count),
            nn.Sigmoid(),
        )
        self.nn_trains = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(train_count),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        data = data.clone()

        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        data.x = self.g1(data)
        data.x = self.g2(data)

        g = global_mean_pool(data.x, batch=data.batch)

        x = self.nn(g)
        switch_states = self.nn_switches(x)
        train_states = self.nn_trains(x)

        return switch_states.reshape(-1), train_states.reshape(-1)
