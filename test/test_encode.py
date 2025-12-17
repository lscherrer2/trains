from unittest import TestCase
from trains.env import System
import networkx as nx
import numpy as np

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

    def test_encode(self):
        G = self.system.encode()

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
        G = self.system.encode(edge_subdivisions=10)

        self.assertEqual(G.nodes[0]["x"].shape, (3,))
        self.assertEqual(G.nodes[1]["x"].shape, (3,))
        self.assertEqual(G.nodes[2]["x"].shape, (3,))
        self.assertEqual(G.nodes[3]["x"].shape, (3,))

        self.assertAlmostEqual(float(G.nodes[0]["x"][0]), 0.0, 6)
        self.assertAlmostEqual(float(G.nodes[2]["x"][0]), 0.0, 6)

        np.testing.assert_allclose(G.nodes[0]["x"], G.nodes[1]["x"])
        np.testing.assert_allclose(G.nodes[2]["x"], G.nodes[3]["x"])

        np.testing.assert_allclose(
            G.nodes[0]["x"][1:], np.array([1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            G.nodes[2]["x"][1:], np.array([0.0, 1.0], dtype=np.float32)
        )

    def test_encode_overlap(self):
        n = 10
        G = self.system.encode(edge_subdivisions=n)

        x = G.edges[0, 2]["x"]
        self.assertEqual(x.shape, (2 + n,))

        train = self.system.trains[0]

        exp_a = np.zeros((n,), dtype=np.float32)
        exp_a[: int(np.floor(n * train.head_progress))] = 1.0
        ov_a = G.edges[0, 2]["x"][2:]
        np.testing.assert_allclose(ov_a, exp_a)

        exp_b = np.zeros((n,), dtype=np.float32)
        exp_b[int(np.floor(n * train.tail_progress)) :] = 1.0
        ov_b = G.edges[2, 0]["x"][2:]
        np.testing.assert_allclose(ov_b, exp_b)

        ov_d = G.edges[0, 3]["x"][2:]
        np.testing.assert_allclose(ov_d, np.zeros((n,), dtype=np.float32))

    def test_encode_overlap_full(self):
        train = self.system.trains[0]
        train.length = 100.0
        train.step(5.0)

        n = 10
        G = self.system.encode(edge_subdivisions=n)

        ov = G.edges[0, 2]["x"][2:]
        np.testing.assert_allclose(ov, np.ones((n,), dtype=np.float32))
