from unittest import TestCase
from trains.rl import RLSystemAdapter, Critic, Actor
from trains.env import System
import json


class TestCritic(TestCase):
    def test_critic_shape(self):
        with open("test/data/system.json") as f:
            system_json = json.load(f)

        system = System.from_json(system_json)
        adapter = RLSystemAdapter(
            system,
            branch_segments=10,
            collision_value=-10,
            switch_overlap_value=-1,
            speed_factor=3,
            diverging_factor=1,
            max_steps=None,
        )

        actor = Actor(**adapter.model_info)
        critic = Critic(**adapter.model_info)

        encoding = adapter.encode()
        switch_action, train_action = actor.forward(encoding.data)
        value = critic.forward(encoding.data, switch_action, train_action)
        self.assertEqual(value.shape, (1,))
