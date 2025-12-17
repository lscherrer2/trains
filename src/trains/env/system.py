from __future__ import annotations
from trains.env.components import Switch, Branch, BranchType, Track
from trains.env.entities import Train
from trains.exceptions import SwitchOverlapError, TrainCollisionError
from typing import overload
from itertools import islice
from networkx import DiGraph
from bidict import bidict
from math import floor
from numpy.typing import NDArray
import numpy as np


class System:
    switches: list[Switch]
    trains: list[Train]

    def __init__(self, switches: list[Switch], trains: list[Train]):
        self.switches = switches
        self.trains = trains

    def detect_collisions(self) -> set[tuple[Train, Train, Track]] | None:
        """
        Returns a set of collisions if there are any. Otherwise returns None.
        A collision is of the form (train_a, train_b, track) where trains a and
        b are the trains that collided, and track is the track on which the
        collision occurred.
        """

        class Collision(Exception):
            def __init__(self, a: Train, b: Train, track: Track):
                self.trains = (a, b)
                self.track = track

        def _to_track_pos(track: Track, branch: Branch, progress: float) -> float:
            p = float(np.clip(progress, 0.0, 1.0))
            if branch is track.ends[0]:
                return p
            if branch is track.ends[1]:
                return 1.0 - p
            raise ValueError("Branch is not an endpoint of its track")

        def _interval_on_track(
            track: Track,
            *,
            end_branch: Branch,
            start_progress: float,
            end_progress: float,
        ) -> tuple[float, float]:
            a = _to_track_pos(track, end_branch, start_progress)
            b = _to_track_pos(track, end_branch, end_progress)
            return (a, b) if a <= b else (b, a)

        collisions: set[tuple[Train, Train, Track]] = set()
        for i, train_a in enumerate(self.trains):
            hist_a = list(train_a.history)
            head_a = hist_a[0]
            tail_a = hist_a[-1]
            body_a = hist_a[1:-1]
            hist_a_tracks = list(map(lambda x: x.track, hist_a))
            body_a_tracks = list(map(lambda x: x.track, body_a))

            try:
                for train_b in self.trains[i + 1 :]:
                    hist_b = list(train_b.history)
                    head_b = hist_b[0]
                    tail_b = hist_b[-1]
                    body_b = hist_b[1:-1]
                    hist_b_tracks = list(map(lambda x: x.track, hist_b))
                    body_b_tracks = list(map(lambda x: x.track, body_b))

                    # Body segments take up the full track.
                    for body_a_track in body_a_tracks:
                        if body_a_track in hist_b_tracks:
                            raise Collision(
                                train_a,
                                train_b,
                                body_a_track,
                            )

                    for body_b_track in body_b_tracks:
                        if body_b_track in hist_a_tracks:
                            raise Collision(
                                train_a,
                                train_b,
                                body_b_track,
                            )

                    # Check head-head collision
                    if head_a.track is head_b.track:
                        a_start = (
                            train_a.tail_progress
                            if tail_a.track is head_a.track
                            else 0.0
                        )
                        a_end = train_a.head_progress
                        b_start = (
                            train_b.tail_progress
                            if tail_b.track is head_b.track
                            else 0.0
                        )
                        b_end = train_b.head_progress

                        a0, a1 = _interval_on_track(
                            head_a.track,
                            end_branch=head_a,
                            start_progress=a_start,
                            end_progress=a_end,
                        )
                        b0, b1 = _interval_on_track(
                            head_b.track,
                            end_branch=head_b,
                            start_progress=b_start,
                            end_progress=b_end,
                        )

                        if max(a0, b0) <= min(a1, b1):
                            raise Collision(
                                train_a,
                                train_b,
                                head_a.track,
                            )

                    # Check tail-tail collision
                    if tail_a.track is tail_b.track:
                        a_end = (
                            train_a.head_progress
                            if head_a.track is tail_a.track
                            else 1.0
                        )
                        b_end = (
                            train_b.head_progress
                            if head_b.track is tail_b.track
                            else 1.0
                        )

                        a0, a1 = _interval_on_track(
                            tail_a.track,
                            end_branch=tail_a,
                            start_progress=train_a.tail_progress,
                            end_progress=a_end,
                        )
                        b0, b1 = _interval_on_track(
                            tail_b.track,
                            end_branch=tail_b,
                            start_progress=train_b.tail_progress,
                            end_progress=b_end,
                        )

                        if max(a0, b0) <= min(a1, b1):
                            raise Collision(
                                train_a,
                                train_b,
                                tail_a.track,
                            )

            except Collision as e:
                collisions.add((*e.trains, e.track))

        return collisions or None

    def step(self, dt: float) -> System:
        for train in self.trains:
            train.step(dt)

        collisions = self.detect_collisions()
        if collisions is not None:
            raise TrainCollisionError(collisions)

        return self

    @overload
    def set_switch_state(self, switch: Switch, state: bool): ...

    @overload
    def set_switch_state(self, switch: str | int, state: bool): ...

    def set_switch_state(self, switch: Switch | str | int, state: bool):
        """Update the state of a switch to switched, not switched

        Args:
            switch (Switch | str | int): The switch to update state for
            state (bool): True maps the switch to diverging, False to through.

        Raises:
            TrainOverlapError: If a train overlaps the switch it will raise this
            exception.
            ValueError: If a train overlaps the switch it will raise this
            exception.
        """
        if isinstance(switch, (str, int)):
            switch = self.switch_map[switch]

        # Check that no train overlaps switch
        if self.is_switch_overlapped(switch):
            raise SwitchOverlapError("Cannot flip switch while train overlaps")

        switch.state = state

    def is_switch_overlapped(self, switch: Switch | str | int) -> bool:
        if isinstance(switch, (str, int)):
            switch = self.switch_map[switch]

        for train in self.trains:
            train.trim()
            for b in islice(train.history, 1, None):
                if b.parent is switch:
                    return True

        return False

    @property
    def switch_map(self) -> dict[int | str, Switch]:
        return {s.tag: s for s in self.switches if s.tag}

    @switch_map.setter
    def switch_map(self, _):
        raise AttributeError("`switch_map` is read-only")

    @property
    def train_map(self) -> dict[int | str, Train]:
        return {t.tag: t for t in self.trains if t.tag}

    @train_map.setter
    def train_map(self, _):
        raise AttributeError("`train_map` is read-only")

    @classmethod
    def from_json(cls, json: dict) -> System:
        from trains.serialization.util import system_from_json

        return system_from_json(json)

    def encode(self, edge_subdivisions: int = 10) -> DiGraph:
        G = DiGraph()

        f_switch_map = bidict()
        b_switch_map = bidict()

        for i, switch in enumerate(self.switches):
            f_switch_map |= {switch: 2 * i}
            b_switch_map |= {switch: 2 * i + 1}

        # Encode forward and backward switches
        for i, switch in enumerate(self.switches):
            # Orthogonal vector encoding unique to each switch
            orth = np.zeros((len(self.switches),), dtype=np.float32)
            orth[i] = 1.0

            # Concatenate with switch encoding state
            data = np.concatenate((switch.encode(), orth))

            # Add nodes for forward and backward switch
            G.add_node(f_switch_map[switch], x=data)
            G.add_node(b_switch_map[switch], x=data)

        # Encode and add edges
        for switch in self.switches:
            for branch in (switch.approach, switch.through, switch.diverging):
                from_ = branch
                to = branch.to()

                from_node = (
                    b_switch_map[from_.parent]
                    if from_.type_ is BranchType.APPROACH
                    else f_switch_map[from_.parent]
                )
                to_node = (
                    f_switch_map[to.parent]
                    if to.type_ is BranchType.APPROACH
                    else b_switch_map[to.parent]
                )

                edge_data = np.concatenate(
                    (branch.encode(), self.encode_overlap(branch, edge_subdivisions))
                )
                G.add_edge(from_node, to_node, x=edge_data)

        return G

    def encode_overlap(self, branch: Branch, segments: int) -> NDArray[np.float32]:  # type: ignore
        overlap = np.zeros((segments,), dtype=np.float32)
        for train in self.trains:
            train.trim()

            hist = train.history
            head = hist[0]
            tail = hist[-1]

            hs = int(floor(segments * float(np.clip(train.head_progress, 0.0, 1.0))))
            ts = int(floor(segments * float(np.clip(train.tail_progress, 0.0, 1.0))))

            if len(hist) >= 3 and any(
                b is branch for b in islice(hist, 1, len(hist) - 1)
            ):
                overlap[:] = 1.0
                break

            if len(hist) == 1 and branch is head and branch is tail:
                start = min(ts, hs)
                end = max(ts, hs)
                overlap[start:end] = 1.0
                continue

            if branch is head:
                overlap[:hs] = 1.0

            if branch is tail:
                overlap[ts:] = 1.0

        return overlap
