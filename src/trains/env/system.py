from __future__ import annotations

from typing import Any, Iterable

from trains.env.deadend import DeadEnd
from trains.env.switch import Switch
from trains.env.track import Track
from trains.env.train import Train
from trains.exceptions import SwitchOverlapError, TrainCollisionError
from trains.ser.system import (
    BranchModel,
    DeadEndBranchModel,
)


class System:
    def __init__(
        self,
        switches: Iterable[Switch],
        deadends: Iterable[DeadEnd],
        trains: Iterable[Train],
    ):
        self.switches = list(switches)
        self.deadends = list(deadends)
        self.trains = list(trains)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> System:
        from trains.ser.system import SystemModel

        model = SystemModel(**data)

        switches: dict[str, Switch] = {}
        for switch_model in model.switches:
            switch = Switch(tag=switch_model.tag, state=switch_model.state)
            switches |= {switch.tag: switch}

        deadends: dict[str, DeadEnd] = {}
        for end_model in model.deadends:
            end = DeadEnd(tag=end_model.tag)
            deadends |= {end.tag: end}

        def resolve_branch(bmodel: BranchModel):
            if isinstance(bmodel, DeadEndBranchModel):
                return deadends[bmodel.node].branch
            else:
                return switches[bmodel.node].get_branch(bmodel.branch)

        tracks = []
        for track_model in model.tracks:
            from_branch = resolve_branch(track_model.from_)
            to_branch = resolve_branch(track_model.to)
            track = Track(
                ends=(from_branch, to_branch),
                length=track_model.length,
            )
            from_branch.track = track
            to_branch.track = track
            tracks.append(track)

        trains = {}
        for train_model in model.trains:
            train = Train(
                tag=train_model.tag,
                speed=train_model.speed,
                length=train_model.length,
                head_distance=train_model.head_distance,
                head_branch=resolve_branch(train_model.head_branch),
            )
            trains |= {train.tag: train}

        return cls(
            switches=switches.values(),
            deadends=deadends.values(),
            trains=trains.values(),
        )

    def step(self, dt: float):
        collisions = self.detect_collisions()
        if collisions:
            raise TrainCollisionError(collisions)

        for train in self.trains:
            train.step(dt)

        if collisions := self.detect_collisions():
            raise TrainCollisionError(collisions)

    def set_switch_state(self, switch_tag: str | int, state: bool):
        switch = self.node_map[switch_tag]

        overlapping_trains = []
        for train in self.trains:
            if self._train_overlaps_switch(train, switch):
                overlapping_trains.append(train)

        if overlapping_trains:
            raise SwitchOverlapError(switch, overlapping_trains)

        switch.state = state

    def _train_overlaps_switch(self, train: Train, switch: Switch) -> bool:
        for branch in switch.branches:
            if branch in train.history:
                return True
        return False

    def detect_collisions(self) -> list[tuple[Train, Train, Track]] | None:
        collisions = []

        track_trains: dict[Track, list[Train]] = {}
        for train in self.trains:
            occupied_tracks = self._get_occupied_tracks(train)
            for track in occupied_tracks:
                if track not in track_trains:
                    track_trains[track] = []
                track_trains[track].append(train)

        for track, trains_on_track in track_trains.items():
            if len(trains_on_track) < 2:
                continue

            for i, train_a in enumerate(trains_on_track):
                for train_b in trains_on_track[i + 1 :]:
                    if self._trains_collide_on_track(train_a, train_b, track):
                        collisions.append((train_a, train_b, track))

        return collisions if collisions else None

    def _get_occupied_tracks(self, train: Train) -> set[Track]:
        tracks = set()
        distance_covered = 0.0

        for i, branch in enumerate(train.history):
            if branch.track is None:
                break

            tracks.add(branch.track)

            if i == 0:
                distance_on_branch = train.head_distance
            else:
                distance_on_branch = branch.track.length

            distance_covered += distance_on_branch

            if distance_covered >= train.length:
                break

        return tracks

    def _trains_collide_on_track(
        self, train_a: Train, train_b: Train, track: Track
    ) -> bool:
        pos_a = self._get_train_position_on_track(train_a, track)
        pos_b = self._get_train_position_on_track(train_b, track)

        if pos_a is None or pos_b is None:
            return False

        start_a, end_a = pos_a
        start_b, end_b = pos_b

        return not (end_a < start_b or end_b < start_a)

    def _get_train_position_on_track(
        self, train: Train, track: Track
    ) -> tuple[float, float] | None:
        distance_covered = 0.0

        for i, branch in enumerate(train.history):
            if branch.track is None:
                break

            if i == 0:
                distance_on_branch = train.head_distance
            else:
                distance_on_branch = branch.track.length

            if branch.track is track:
                if i == 0:
                    head_pos = train.head_distance
                    tail_pos = max(0.0, head_pos - train.length)
                else:
                    remaining_length = train.length - distance_covered
                    head_pos = branch.track.length
                    tail_pos = branch.track.length - remaining_length

                return (min(tail_pos, head_pos), max(tail_pos, head_pos))

            distance_covered += distance_on_branch

            if distance_covered >= train.length:
                break

        return None

    @property
    def nodes(self):
        return self.switches + self.deadends

    @property
    def node_map(self):
        return self.switch_map | self.deadend_map

    @property
    def switch_map(self):
        return {s.tag: s for s in self.switches}

    @property
    def deadend_map(self):
        return {d.tag: d for d in self.deadends}

    @property
    def train_map(self):
        return {t.tag: t for t in self.trains}
