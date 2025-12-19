from unittest import TestCase

import json
import torch

from trains.env import System
from trains.rl import RLSystemAdapter
from trains.rl.adapter import Action


class TestRLSystemAdapter(TestCase):
    def test_step_sets_switches_and_speeds(self):
        with open("test/data/system.json") as f:
            graph_json = json.load(f)

        system = System.from_json(graph_json)
        adapter = RLSystemAdapter(
            system,
            branch_segments=10,
            collision_value=-20,
            switch_overlap_value=-1,
            speed_factor=5,
            diverging_factor=3,
        )

        # system.json contains two switches (A, B) and one train.
        # The train overlaps switch B, so we do not attempt to flip it.
        # We do flip switch A using a probability-like tensor.
        switch_states = torch.tensor([0.9, 0.1], dtype=torch.float32)
        train_speeds = torch.tensor([0.5], dtype=torch.float32)
        action = Action(switch_states=switch_states, train_speeds=train_speeds)

        adapter.step(action, dt=0.0)

        self.assertIs(system.switches[0].state, True)
        self.assertIs(system.switches[1].state, False)

        self.assertIsInstance(system.trains[0].speed, float)
        self.assertEqual(system.trains[0].speed, 0.5)
