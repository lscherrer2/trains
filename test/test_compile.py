from unittest import TestCase
from trains.env import System
import networkx as nx
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def curved_edges(G, pos, rad=0.2):
    ax = plt.gca()
    for u, v in G.edges():
        patch = FancyArrowPatch(
            pos[u], pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle='-|>',
            mutation_scale=15,
            lw=1.5,
            color='black'
        )
        ax.add_patch(patch)


def draw_graph_curved(G):
    pos = nx.spring_layout(G)

    # Draw nodes + labels ONLY (no edges!)
    nx.draw_networkx_nodes(G, pos, node_color='#6BA3D6')
    nx.draw_networkx_labels(G, pos)

    # Draw ALL edges curved
    curved_edges(G, pos, rad=0.25)

    plt.axis("off")
    plt.show()

class TestEncode(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.system = System.from_json(graph_json)

    # TODO: Fix
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
