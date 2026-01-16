import json
from unittest import TestCase

from trains.env import System


class TestJsonInit(TestCase):
    def setUp(self):
        with open("test/data/init_system.json") as f:
            graph_json = json.load(f)

        self.G = System.from_json(graph_json)

    def test_components(self):
        self.assertEqual(len(self.G.switches), 1)
        self.assertEqual(len(self.G.deadends), 3)
        self.assertEqual(len(self.G.trains), 1)

        self.assertIn("A", self.G.switch_map)
        self.assertIn("T", self.G.train_map)
        for b in ["approach", "through", "diverge"]:
            self.assertIn(b, self.G.deadend_map)

    def test_trains(self):
        self.assertIn("T", self.G.train_map)

        train = self.G.train_map["T"]
        self.assertEqual(train.tag, "T")
        self.assertEqual(train.speed, 1.0)
        self.assertEqual(train.length, 10.0)
        self.assertEqual(train.head_distance, 5.0)

        target_branch = self.G.switch_map["A"].through
        self.assertIs(train.head_branch, target_branch)

    def test_tracks(self):
        switch = self.G.switch_map["A"]
        branches = {
            "through": switch.through.track,
            "diverge": switch.diverge.track,
            "approach": switch.approach.track,
        }
        deadends = {
            "approach": self.G.deadend_map["approach"].branch.track,
            "through": self.G.deadend_map["through"].branch.track,
            "diverge": self.G.deadend_map["diverge"].branch.track,
        }
        for b in ["approach", "through", "diverge"]:
            self.assertIs(branches[b], deadends[b])
