from unittest import TestCase
from trains.rl import Actor, RLSystemAdapter
from trains.env import System
import json
import torch


class TestActor(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.system = System.from_json(graph_json)
        adapter = RLSystemAdapter(
            self.system,
            branch_segments=10,
            collision_value=-20,
            switch_overlap_value=-1,
            speed_factor=5,
            diverging_factor=3,
        )

        self.encoding = adapter.encode()
        x = self.encoding.data.x
        edge_attr = self.encoding.data.edge_attr

        node_dim = int(x.shape[1])
        edge_dim = int(edge_attr.shape[1]) if edge_attr.dim() > 1 else 1

        switch_count = len(self.system.switches)
        train_count = len(self.system.trains)

        self.actor = Actor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            switch_count=switch_count,
            train_count=train_count,
        )

    def test_forward_shapes(self):
        switch_states, train_states = self.actor(self.encoding.data)

        self.assertEqual(tuple(switch_states.shape), (len(self.system.switches),))
        self.assertEqual(tuple(train_states.shape), (len(self.system.trains),))
        self.assertTrue(torch.isfinite(switch_states).all().item())
        self.assertTrue(torch.isfinite(train_states).all().item())
