from unittest import TestCase
from trains.env import System
import networkx as nx
import json


class TestEncode(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.system = System.from_json(graph_json)

    # # TODO: Fix
    # def test_encode(self):
    #     G = self.system.encode()
    #     import matplotlib.pyplot as plt
    #
    #     nx.draw_networkx(
    #         G,
    #         nx.spring_layout(G),
    #         with_labels=True,
    #         arrows=True,
    #         arrowstyle="->",
    #         arrowsize=10,
    #     )
    #     plt.show()
