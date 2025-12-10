from unittest import TestCase
from trains.env import System
import json


class TestSim(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.G = System.from_json(graph_json)
