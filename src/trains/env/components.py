from __future__ import annotations
from enum import Enum


class Receptor(Enum):
    APPROACH = 1
    THROUGH = 2
    DIVERGING = 3


class Switch:
    approach: TrackSegment
    through: TrackSegment
    diverging: TrackSegment
    state: bool

    def embed(self, trains: list[Train], direction) -> list[float]:
        beneath_train = any(map(lambda train: self in train.covered, trains))

        return [
            float(self.state),  # 1.0 if switched
            float(beneath_train),  # If train is overlapping
        ]


class Train:
    covered: list[Switch]


class TrackSegment:
    ends: tuple[tuple[Switch, Receptor], tuple[Switch, Receptor]]

    def other(self, switch: Switch) -> Switch:
        assert switch in self.ends, "switch must be in ends"
        return self.ends[0][0] if self.ends[1][0] == switch else self.ends[1][0]
