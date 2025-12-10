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