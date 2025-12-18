from unittest import TestCase
import json

from trains.env import System
from trains.exceptions import SwitchOverlapError, TrainCollisionError


def _load_system(path: str) -> System:
    with open(path) as f:
        return System.from_json(json.load(f))


def _norm(collisions):
    if not collisions:
        return set()
    return {(frozenset((a.tag, b.tag)), t) for (a, b, t) in collisions}


class TestCollisions(TestCase):
    def test_fixture_collision_system_expected_pairs(self):
        G = _load_system("test/data/collision_system.json")
        collisions = G.detect_collisions()
        self.assertIsNotNone(collisions)

        track = G.switch_map["A"].through.track
        norm = _norm(collisions)

        self.assertIn((frozenset(("T1", "T3")), track), norm)
        self.assertIn((frozenset(("TA", "TB")), track), norm)
        self.assertNotIn((frozenset(("T1", "T2")), track), norm)

        self.assertEqual(len(norm), 2)

    def test_opposite_direction_same_track_no_overlap(self):
        G = _load_system("test/data/collision_system.json")

        ta = G.train_map["TA"]
        tb = G.train_map["TB"]
        G.trains = [ta, tb]

        ta.length = 1.0
        tb.length = 1.0
        ta.head_progress = 0.1
        tb.head_progress = 0.1
        ta.trim()
        tb.trim()

        self.assertIsNone(G.detect_collisions())

    def test_boundary_touch_counts_as_collision(self):
        G = _load_system("test/data/collision_touch_system.json")
        collisions = G.detect_collisions()
        self.assertIsNotNone(collisions)

        track = G.switch_map["A"].through.track
        norm = _norm(collisions)

        self.assertIn((frozenset(("T1", "T2")), track), norm)
        self.assertEqual(len(norm), 1)

    def test_body_segment_occupies_entire_track(self):
        G = _load_system("test/data/collision_body_system.json")
        collisions = G.detect_collisions()
        self.assertIsNotNone(collisions)

        body_track = G.switch_map["B"].through.track
        norm = _norm(collisions)

        self.assertIn((frozenset(("TL", "TS")), body_track), norm)
        self.assertEqual(len(norm), 1)

    def test_step_collision(self):
        G = _load_system("test/data/step_collision_system.json")

        for _ in range(20):
            try:
                G.step(0.5)
            except TrainCollisionError as e:
                track = G.switch_map["A"].through.track
                norm = _norm(e.collisions)
                self.assertIn((frozenset(("TA", "TB")), track), norm)
                return

        self.fail("Expected TrainCollisionError, but no collision occurred")

    def test_switch_overlap(self):
        G = _load_system("test/data/system.json")
        with self.assertRaises(SwitchOverlapError):
            G.set_switch_state("B", True)
