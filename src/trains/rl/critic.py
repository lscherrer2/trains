from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from trains.rl.util import GNN
from torch import Tensor
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        switch_count: int,
        train_count: int,
    ):
        super().__init__()

        GRAPH_DIM = 64
        self.g1 = GNN(in_dim=node_dim, out_dim=GRAPH_DIM, edge_dim=edge_dim)
        self.g2 = GNN(in_dim=GRAPH_DIM, out_dim=GRAPH_DIM, edge_dim=edge_dim)

        self.nn = nn.Sequential(
            nn.Linear(GRAPH_DIM + switch_count + train_count, 128),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

    def forward(
        self,
        state: Data,
        switch_action: Tensor,
        train_action: Tensor,
    ) -> Tensor:
        state = state.clone()

        state.batch = torch.zeros(
            state.num_nodes,
            dtype=torch.long,
        )

        state.x = self.g1(state)
        state.x = self.g2(state)

        g = global_mean_pool(state.x, batch=state.batch)

        x = torch.cat((g.reshape(-1), switch_action, train_action))
        x: Tensor = self.nn(x)

        return x.reshape(-1)
