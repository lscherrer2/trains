from unittest import TestCase
from trains.env import System
from trains.rl import RLSystemAdapter
import networkx as nx
import numpy as np
import torch

import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def curved_edges(G, pos, rad=0.2):
    ax = plt.gca()
    for u, v in G.edges():
        patch = FancyArrowPatch(
            pos[u],
            pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=1.5,
            color="black",
        )
        ax.add_patch(patch)


def draw_graph_curved(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="#6BA3D6")
    nx.draw_networkx_labels(G, pos)
    curved_edges(G, pos, rad=0.25)
    plt.axis("off")
    plt.show()


class TestEncode(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.system = System.from_json(graph_json)
        self.adapter = RLSystemAdapter(
            self.system,
            branch_segments=10,
            collision_value=-20,
            switch_overlap_value=-1,
            speed_factor=5,
            diverging_factor=3,
        )

    def test_encode(self):
        encoding = self.adapter.encode()
        G = encoding.nx_graph

        ## Uncomment below to view the graph
        # draw_graph_curved(G)

        # Number of switches * 2
        self.assertEqual(G.number_of_nodes(), 4)

        self.assertEqual(G.number_of_edges(0, 0), 0)
        self.assertEqual(G.number_of_edges(0, 1), 0)
        self.assertEqual(G.number_of_edges(0, 2), 1)
        self.assertEqual(G.number_of_edges(0, 3), 1)

        self.assertEqual(G.number_of_edges(1, 0), 0)
        self.assertEqual(G.number_of_edges(1, 1), 0)
        self.assertEqual(G.number_of_edges(1, 2), 0)
        self.assertEqual(G.number_of_edges(1, 3), 1)

        self.assertEqual(G.number_of_edges(2, 0), 1)
        self.assertEqual(G.number_of_edges(2, 1), 1)
        self.assertEqual(G.number_of_edges(2, 2), 0)
        self.assertEqual(G.number_of_edges(2, 3), 0)

        self.assertEqual(G.number_of_edges(3, 0), 0)
        self.assertEqual(G.number_of_edges(3, 1), 1)
        self.assertEqual(G.number_of_edges(3, 2), 0)
        self.assertEqual(G.number_of_edges(3, 3), 0)

    def test_node_embeddings(self):
        encoding = self.adapter.encode()
        G = encoding.nx_graph

        self.assertEqual(tuple(G.nodes[0]["x"].shape), (3,))
        self.assertEqual(tuple(G.nodes[1]["x"].shape), (3,))
        self.assertEqual(tuple(G.nodes[2]["x"].shape), (3,))
        self.assertEqual(tuple(G.nodes[3]["x"].shape), (3,))

        self.assertAlmostEqual(float(G.nodes[0]["x"][0]), 0.0, 6)
        self.assertAlmostEqual(float(G.nodes[2]["x"][0]), 0.0, 6)

        torch.testing.assert_close(G.nodes[0]["x"], G.nodes[1]["x"])
        torch.testing.assert_close(G.nodes[2]["x"], G.nodes[3]["x"])

        torch.testing.assert_close(G.nodes[0]["x"][1:], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(G.nodes[2]["x"][1:], torch.tensor([0.0, 1.0]))

    def test_encode_overlap(self):
        n = 10
        encoding = self.adapter.encode()
        G = encoding.nx_graph

        x = G.edges[0, 2]["edge_attr"]
        self.assertEqual(tuple(x.shape), (2 + n,))

        train = self.system.trains[0]

        exp_a = np.zeros((n,), dtype=np.float32)
        exp_a[: int(np.floor(n * train.head_progress))] = 1.0
        ov_a = G.edges[0, 2]["edge_attr"][2:].detach().cpu().numpy()
        np.testing.assert_allclose(ov_a, exp_a)

        exp_b = np.zeros((n,), dtype=np.float32)
        exp_b[int(np.floor(n * train.tail_progress)) :] = 1.0
        ov_b = G.edges[2, 0]["edge_attr"][2:].detach().cpu().numpy()
        np.testing.assert_allclose(ov_b, exp_b)

        ov_d = G.edges[0, 3]["edge_attr"][2:].detach().cpu().numpy()
        np.testing.assert_allclose(ov_d, np.zeros((n,), dtype=np.float32))

    def test_encode_overlap_full(self):
        train = self.system.trains[0]
        train.length = 100.0
        train.step(5.0)

        n = 10
        encoding = self.adapter.encode()
        G = encoding.nx_graph

        ov = G.edges[0, 2]["edge_attr"][2:].detach().cpu().numpy()
        np.testing.assert_allclose(ov, np.ones((n,), dtype=np.float32))
