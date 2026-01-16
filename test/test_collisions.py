from unittest import TestCase

from trains.env import System
from trains.exceptions import TrainCollisionError


def make_simple_system(trains_data):
    json_data = {
        "switches": [],
        "deadends": [{"tag": "A"}, {"tag": "B"}],
        "tracks": [
            {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0}
        ],
        "trains": trains_data,
    }
    return System.from_json(json_data)


class TestCollisions(TestCase):
    def setUp(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 5.0,
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 1.0,
                "length": 1.0,
                "head_distance": 2.0,
                "head_branch": {"node": "A"},
            },
        ]
        self.G = make_simple_system(trains)

    def test_initial_state_no_collision(self):
        collisions = self.G.detect_collisions()
        self.assertIsNone(collisions)

    def test_detect_collisions_returns_none_when_no_collision(self):
        collisions = self.G.detect_collisions()
        self.assertIsNone(collisions)

    def test_get_occupied_tracks(self):
        train = self.G.train_map["T1"]
        tracks = self.G._get_occupied_tracks(train)
        self.assertEqual(len(tracks), 1)

    def test_both_trains_on_same_track(self):
        t1 = self.G.train_map["T1"]
        t2 = self.G.train_map["T2"]

        tracks_t1 = self.G._get_occupied_tracks(t1)
        tracks_t2 = self.G._get_occupied_tracks(t2)

        self.assertEqual(tracks_t1, tracks_t2)


class TestCollisionAfterMovement(TestCase):
    def setUp(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 5.0,
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 1.0,
                "length": 1.0,
                "head_distance": 2.0,
                "head_branch": {"node": "A"},
            },
        ]
        self.G = make_simple_system(trains)

    def test_step_raises_collision_when_trains_meet(self):
        self.G.step(1.0)

        with self.assertRaises(TrainCollisionError):
            self.G.step(1.0)


class TestTrainPositions(TestCase):
    def setUp(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 5.0,
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 1.0,
                "length": 1.0,
                "head_distance": 2.0,
                "head_branch": {"node": "A"},
            },
        ]
        self.G = make_simple_system(trains)

    def test_train_head_distance(self):
        t1 = self.G.train_map["T1"]
        t2 = self.G.train_map["T2"]

        self.assertAlmostEqual(t1.head_distance, 5.0)
        self.assertAlmostEqual(t2.head_distance, 2.0)

    def test_train_length(self):
        t1 = self.G.train_map["T1"]
        t2 = self.G.train_map["T2"]

        self.assertAlmostEqual(t1.length, 1.0)
        self.assertAlmostEqual(t2.length, 1.0)

    def test_train_speed(self):
        t1 = self.G.train_map["T1"]
        t2 = self.G.train_map["T2"]

        self.assertAlmostEqual(t1.speed, 0.0)  # Stationary
        self.assertAlmostEqual(t2.speed, 1.0)  # Moving

    def test_train_moves_correctly(self):
        t2 = self.G.train_map["T2"]
        initial_pos = t2.head_distance

        t2.step(1.0)

        self.assertAlmostEqual(t2.head_distance, initial_pos + 1.0)


class TestHeadOnCollision(TestCase):
    """Test head-on collisions with trains moving toward each other."""

    def test_head_on_collision_initial_state(self):
        """Test that trains moving toward each other start without collision."""
        trains = [
            {
                "tag": "T1",
                "speed": -1.0,  # Moving backward (toward A)
                "length": 1.0,
                "head_distance": 7.0,
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 1.0,  # Moving forward (toward B)
                "length": 1.0,
                "head_distance": 3.0,
                "head_branch": {"node": "A"},
            },
        ]
        G = make_simple_system(trains)

        self.assertIsNone(G.detect_collisions())


class TestNoTrainCollisions(TestCase):
    def test_single_train_no_collision(self):
        trains = [
            {
                "tag": "T1",
                "speed": 1.0,
                "length": 1.0,
                "head_distance": 5.0,
                "head_branch": {"node": "A"},
            }
        ]
        G = make_simple_system(trains)

        collisions = G.detect_collisions()
        self.assertIsNone(collisions)

    def test_trains_on_different_tracks_no_collision(self):
        json_data = {
            "switches": [],
            "deadends": [
                {"tag": "A"},
                {"tag": "B"},
                {"tag": "C"},
                {"tag": "D"},
            ],
            "tracks": [
                {"from_": {"node": "A"}, "to": {"node": "B"}, "length": 10.0},
                {"from_": {"node": "C"}, "to": {"node": "D"}, "length": 10.0},
            ],
            "trains": [
                {
                    "tag": "T1",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "A"},
                },
                {
                    "tag": "T2",
                    "speed": 0.0,
                    "length": 1.0,
                    "head_distance": 5.0,
                    "head_branch": {"node": "C"},
                },
            ],
        }
        G = System.from_json(json_data)

        collisions = G.detect_collisions()
        self.assertIsNone(collisions)


class TestOverlappingTrains(TestCase):
    def test_overlapping_trains_detected(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 5.0,  # Occupies [3, 5]
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 4.0,  # Occupies [2, 4] - overlaps with T1 at [3, 4]
                "head_branch": {"node": "A"},
            },
        ]
        G = make_simple_system(trains)

        collisions = G.detect_collisions()
        self.assertIsNotNone(collisions)
        self.assertEqual(len(collisions), 1)

    def test_trains_touching_edges(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 5.0,  # Occupies [3, 5]
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 3.0,  # Occupies [1, 3] - touches T1 exactly at 3
                "head_branch": {"node": "A"},
            },
        ]
        G = make_simple_system(trains)

        # Edge touching may or may not be a collision depending on implementation
        # This test documents the behavior
        collisions = G.detect_collisions()
        # Since end_a < start_b or end_b < start_a is the no-collision check,
        # at 3 == 3, neither condition is true, so it should be a collision
        self.assertIsNotNone(collisions)


class TestTrainCollisionError(TestCase):
    def test_collision_error_contains_train_info(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 5.0,
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 4.0,  # Overlaps with T1
                "head_branch": {"node": "A"},
            },
        ]
        G = make_simple_system(trains)

        with self.assertRaises(TrainCollisionError) as ctx:
            G.step(0.1)

        error = ctx.exception
        self.assertIsNotNone(error.trains)
        self.assertEqual(len(error.trains), 1)

        collision_tuple = error.trains[0]
        self.assertEqual(len(collision_tuple), 3)  # (train_a, train_b, track)


class TestMultipleTrainCollisions(TestCase):
    def test_three_trains_two_collisions(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 3.0,  # Occupies [2, 3]
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 0.0,
                "length": 2.0,
                "head_distance": 4.0,  # Occupies [2, 4] - overlaps with T1
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T3",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 3.5,  # Occupies [2.5, 3.5] - overlaps with T2
                "head_branch": {"node": "A"},
            },
        ]
        G = make_simple_system(trains)

        collisions = G.detect_collisions()
        self.assertIsNotNone(collisions)
        self.assertGreaterEqual(len(collisions), 1)

    def test_no_collision_with_separated_trains(self):
        trains = [
            {
                "tag": "T1",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 2.0,  # Occupies [1, 2]
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T2",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 5.0,  # Occupies [4, 5]
                "head_branch": {"node": "A"},
            },
            {
                "tag": "T3",
                "speed": 0.0,
                "length": 1.0,
                "head_distance": 8.0,  # Occupies [7, 8]
                "head_branch": {"node": "A"},
            },
        ]
        G = make_simple_system(trains)

        collisions = G.detect_collisions()
        self.assertIsNone(collisions)
