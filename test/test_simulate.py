import json
from unittest import TestCase

from trains.env import System
from trains.env.deadend import DeadEndCollision
from trains.env.switch import SwitchPassthroughError
from trains.exceptions import TrainCollisionError, SwitchOverlapError


class TestTrainMovement(TestCase):
    def setUp(self):
        with open("test/data/simulate_system.json") as f:
            self.json_data = json.load(f)
        self.G = System.from_json(self.json_data)

    def test_train_moves_forward(self):
        train = self.G.train_map["T1"]
        initial_pos = train.head_distance

        self.G.step(1.0)

        self.assertAlmostEqual(train.head_distance, initial_pos + 1.0)

    def test_train_moves_multiple_steps(self):
        train = self.G.train_map["T1"]

        for i in range(4):
            self.G.step(1.0)

        self.assertAlmostEqual(train.head_distance, 9.0)

    def test_train_crosses_to_next_track(self):
        train = self.G.train_map["T1"]

        self.G.step(4.0)
        self.assertAlmostEqual(train.head_distance, 9.0)

        self.G.step(2.0)
        self.assertIs(train.head_branch.parent, self.G.switch_map["S1"])
        self.assertAlmostEqual(train.head_distance, 1.0)

    def test_train_history_grows_when_crossing_tracks(self):
        train = self.G.train_map["T1"]
        initial_history_len = len(train.history)

        self.G.step(6.0)

        self.assertGreater(len(train.history), initial_history_len)

    def test_stationary_train_does_not_move(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]

        G.step(10.0)

        self.assertAlmostEqual(train.head_distance, 5.0)


class TestSwitchTraversal(TestCase):
    def test_train_goes_through_when_switch_state_false(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D1"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]
        switch = G.switch_map["S1"]

        G.step(2.0)

        self.assertIs(train.head_branch, switch.through)

    def test_train_diverges_when_switch_state_true(self):
        json_data = {
            "switches": [{"tag": "S1", "state": True}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D1"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]
        switch = G.switch_map["S1"]

        G.step(2.0)

        self.assertIs(train.head_branch, switch.diverge)

    def test_train_passes_through_multiple_switches(self):
        with open("test/data/simulate_system.json") as f:
            json_data = json.load(f)
        G = System.from_json(json_data)
        train = G.train_map["T1"]
        s2 = G.switch_map["S2"]

        G.step(16.0)

        self.assertIs(train.head_branch, s2.through)

    def test_train_from_through_to_approach_when_state_false(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D2"},
                    "to": {"node": "S1", "branch": "through"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D3"},
                    "to": {"node": "S1", "branch": "diverge"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D2"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]
        switch = G.switch_map["S1"]

        G.step(2.0)

        self.assertIs(train.head_branch, switch.approach)

    def test_train_from_diverge_to_approach_when_state_true(self):
        json_data = {
            "switches": [{"tag": "S1", "state": True}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D2"},
                    "to": {"node": "S1", "branch": "through"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D3"},
                    "to": {"node": "S1", "branch": "diverge"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D3"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]
        switch = G.switch_map["S1"]

        G.step(2.0)

        self.assertIs(train.head_branch, switch.approach)

    def test_switch_passthrough_error_wrong_state_from_through(self):
        json_data = {
            "switches": [{"tag": "S1", "state": True}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D2"},
                    "to": {"node": "S1", "branch": "through"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D3"},
                    "to": {"node": "S1", "branch": "diverge"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D2"},
                }
            ],
        }
        G = System.from_json(json_data)

        with self.assertRaises(SwitchPassthroughError):
            G.step(2.0)

    def test_switch_passthrough_error_wrong_state_from_diverge(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D2"},
                    "to": {"node": "S1", "branch": "through"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "D3"},
                    "to": {"node": "S1", "branch": "diverge"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D3"},
                }
            ],
        }
        G = System.from_json(json_data)

        with self.assertRaises(SwitchPassthroughError):
            G.step(2.0)


class TestSwitchStateChanges(TestCase):
    def test_set_switch_state(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 1.0,
                    "head_branch": {"node": "D1"},
                }
            ],
        }
        G = System.from_json(json_data)
        switch = G.switch_map["S1"]

        self.assertFalse(switch.state)

        G.set_switch_state("S1", True)

        self.assertTrue(switch.state)

    def test_switch_state_change_affects_train_path(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 1.0,
                    "head_branch": {"node": "D1"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]
        switch = G.switch_map["S1"]

        G.step(5.0)
        G.set_switch_state("S1", True)
        G.step(5.0)

        self.assertIs(train.head_branch, switch.diverge)

    def test_switch_overlap_error_when_train_on_switch(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "S1", "branch": "through"},
                }
            ],
        }
        G = System.from_json(json_data)

        with self.assertRaises(SwitchOverlapError) as ctx:
            G.set_switch_state("S1", True)

        self.assertEqual(ctx.exception.switch.tag, "S1")
        self.assertEqual(len(ctx.exception.trains), 1)

    def test_switch_can_change_when_train_not_overlapping(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "D1"},
                }
            ],
        }
        G = System.from_json(json_data)
        switch = G.switch_map["S1"]

        G.set_switch_state("S1", True)

        self.assertTrue(switch.state)


class TestDeadEndCollisions(TestCase):
    def test_train_hits_dead_end(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "A"},
                }
            ],
        }
        G = System.from_json(json_data)

        with self.assertRaises(DeadEndCollision) as ctx:
            G.step(2.0)

        self.assertEqual(ctx.exception.dead_end.tag, "B")

    def test_train_stops_at_dead_end_position(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "A"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]

        try:
            G.step(2.0)
        except DeadEndCollision:
            pass

        self.assertAlmostEqual(train.head_distance, 10.0)


class TestTimedCollisions(TestCase):
    def test_collision_at_predictable_time(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 6.0,
                    "head_branch": {"node": "A"},
                },
                {
                    "tag": "T2",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 2.0,
                    "head_branch": {"node": "A"},
                },
            ],
        }
        G = System.from_json(json_data)

        G.step(1.0)
        G.step(1.0)

        with self.assertRaises(TrainCollisionError):
            G.step(1.0)

    def test_head_on_collision(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 20.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "D1"},
                },
                {
                    "tag": "T2",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "S1", "branch": "approach"},
                },
            ],
        }
        G = System.from_json(json_data)

        with self.assertRaises(TrainCollisionError):
            for _ in range(10):
                G.step(1.0)

    def test_no_collision_when_trains_take_different_paths(self):
        json_data = {
            "switches": [{"tag": "S1", "state": True}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 8.0,
                    "head_branch": {"node": "D1"},
                },
                {
                    "tag": "T2",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "S1", "branch": "through"},
                },
            ],
        }
        G = System.from_json(json_data)
        t1 = G.train_map["T1"]
        switch = G.switch_map["S1"]

        G.step(3.0)

        self.assertIs(t1.head_branch, switch.diverge)


class TestTrainTrim(TestCase):
    def test_train_trims_history(self):
        json_data = {
            "switches": [{"tag": "S1", "state": False}],
            "deadends": [{"tag": "D1"}, {"tag": "D2"}, {"tag": "D3"}],
            "tracks": [
                {
                    "from_": {"node": "D1"},
                    "to": {"node": "S1", "branch": "approach"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "through"},
                    "to": {"node": "D2"},
                    "length": 10.0,
                },
                {
                    "from_": {"node": "S1", "branch": "diverge"},
                    "to": {"node": "D3"},
                    "length": 10.0,
                },
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 2.0,
                    "head_distance": 9.0,
                    "head_branch": {"node": "D1"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]

        G.step(5.0)

        train.trim()

        self.assertLessEqual(len(train.history), 2)


class TestMultipleTrains(TestCase):
    def test_multiple_trains_move_independently(self):
        json_data = {
            "switches": [],
            "deadends": [
                {"tag": "A"},
                {"tag": "B"},
                {"tag": "C"},
                {"tag": "D"},
            ],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 20.0},
                {"from_": {"node": "C"}, "to": {"node": "D"}, "length": 20.0},
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                },
                {
                    "tag": "T2",
                    "speed": 2.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "C"},
                },
            ],
        }
        G = System.from_json(json_data)
        t1 = G.train_map["T1"]
        t2 = G.train_map["T2"]

        G.step(5.0)

        self.assertAlmostEqual(t1.head_distance, 10.0)
        self.assertAlmostEqual(t2.head_distance, 15.0)

    def test_trains_on_same_track_different_speeds(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 100.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 2.0,
                    "length": 1.0,
                    "head_distance": 10.0,
                    "head_branch": {"node": "A"},
                },
                {
                    "tag": "T2",
                    "speed": 1.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                },
            ],
        }
        G = System.from_json(json_data)
        t1 = G.train_map["T1"]
        t2 = G.train_map["T2"]

        G.step(1.0)

        self.assertAlmostEqual(t1.head_distance, 12.0)
        self.assertAlmostEqual(t2.head_distance, 6.0)


class TestTrainProperties(TestCase):
    def test_head_progress(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]

        self.assertAlmostEqual(train.head_progress, 0.5)

    def test_set_head_progress(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]

        train.head_progress = 0.8

        self.assertAlmostEqual(train.head_distance, 8.0)

    def test_tail_distance(self):
        json_data = {
            "switches": [],
            "deadends": [{"tag": "A"}, {"tag": "B"}],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 3.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                }
            ],
        }
        G = System.from_json(json_data)
        train = G.train_map["T1"]

        self.assertAlmostEqual(train.tail_distance, 3.0)
