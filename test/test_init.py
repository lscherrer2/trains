from unittest import TestCase, main
from trains.env import System
import json


class TestJsonInit(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.G = System.from_json(graph_json)

    def test_switches(self):
        G = self.G

        self.assertEqual(len(G.switches), 2, "Loaded graph must have 2 switches")
        self.assertIn("A", G.switch_map)
        self.assertIn("B", G.switch_map)

    def test_trains(self):
        G = self.G

        switch_a = G.switch_map["A"]
        switch_b = G.switch_map["B"]

        self.assertEqual(len(G.trains), 1, "Loaded graph must have 1 train")
        self.assertIn("T", G.train_map)
        train = G.train_map["T"]

        self.assertAlmostEqual(train.progress, 0.5, 5)
        self.assertAlmostEqual(train.speed, 1.0, 5)
        self.assertIs(train.history[-1], switch_a.through)

        self.assertEqual(len(train.history), 2)
        self.assertIs(train.history[-1], switch_a.through)
        self.assertIs(train.history[-2], switch_b.through)

        self.assertAlmostEqual(train.tail_progress, 2.0 / 3.0, 5)

    def test_tracks(self):
        G = self.G

        switch_a = G.switch_map["A"]
        switch_b = G.switch_map["B"]

        self.assertIs(switch_a.through.track, switch_b.approach.track)
        self.assertAlmostEqual(switch_a.through.track.length, 10.0, 5)
        self.assertIs(switch_a.diverging.track, switch_b.diverging.track)
        self.assertAlmostEqual(switch_a.diverging.track.length, 20.0, 5)
        self.assertIs(switch_a.approach.track, switch_b.through.track)
        self.assertAlmostEqual(switch_a.approach.track.length, 30.0, 5)
