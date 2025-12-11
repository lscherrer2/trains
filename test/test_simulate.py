from unittest import TestCase
from trains.env import System
import json


class TestSim(TestCase):
    def setUp(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        self.G = System.from_json(graph_json)

    def test_step(self):
        train = self.G.trains[0]
        switch_a = self.G.switch_map["A"]
        switch_b = self.G.switch_map["B"]
        
        self.assertAlmostEqual(train.head_progress, 0.5, 5)
        self.assertAlmostEqual(train.tail_progress, 2.0 / 3.0, 5)

        self.assertAlmostEqual(train.head_distance, 5.0, 5)
        self.assertAlmostEqual(train.tail_distance, 10.0, 5)

        self.assertAlmostEqual(train.speed, 1.0, 5)

        train.step(4.0)

        self.assertAlmostEqual(train.head_distance, 5.0 + 4.0, 5)
        self.assertAlmostEqual(train.tail_distance, 10.0 - 4.0, 5)

        train.step(1.0)

        self.assertAlmostEqual(train.head_distance % 10, 0.0, 5)
        self.assertAlmostEqual(train.tail_distance, 10.0 - 5.0, 5)

        self.assertEqual(len(train.history), 3)

        train.step(5.0)

        self.assertAlmostEqual(train.head_distance % 10, 5.0, 5)
        self.assertAlmostEqual(train.tail_distance % 10, 0.0, 5)

        self.assertIs(train.history[-1], switch_b.through)
        self.assertIs(train.history[-2], switch_a.through)
        self.assertIs(train.history[-3], switch_b.through)
        

    def test_switch(self):
        train = self.G.trains[0]
        switch_b = self.G.switch_map["B"]

        switch_b.state = True # Have it go to diverging

        train.step(6.0)

        self.assertIs(train.history[-1].parent, switch_b)
        self.assertIs(train.history[-1], switch_b.diverging)
        