from trains.env.components import Branch
from collections import deque
from typing import Iterable
import numpy as np


class Train:
    history: deque[Branch]
    head_progress: float
    speed: float
    length: float
    tag: int | str | None

    def __init__(
        self,
        length: float,
        speed: float,
        head_progress: float,
        span: Iterable[Branch],
        tag: int | str | None = None,
    ) -> None:
        self.length = length
        self.speed = speed
        self.head_progress = head_progress
        self.history = deque(span)
        self.tag = tag
        self.trim()

    @property
    def head_distance(self) -> float:
        """Distance the head is from the switch it is departing"""
        return self.head_progress * self.history[0].track.length

    @property
    def tail_distance(self) -> float:
        """Distance the tail is from the switch it is approaching"""
        return (1.0 - self.tail_progress) * self.history[-1].track.length

    def trim(self):
        if not self.history:
            raise ValueError("Train.history cannot be empty")

        head = self.history[0]
        dist = head.track.length * self.head_progress
        hist: deque[Branch] = deque([head])

        for branch in list(self.history)[1:]:
            if dist > self.length:
                break
            hist.append(branch)
            dist += branch.track.length

        self.history = hist

    @property
    def tail_progress(self) -> float:
        self.trim()

        head = self.history[0]
        tail = self.history[-1]

        if len(self.history) == 1:
            L = head.track.length
            if L <= 0.0:
                return 0.0
            return float(np.clip(self.head_progress - (self.length / L), 0.0, 1.0))

        dist = self.head_progress * head.track.length
        for branch in list(self.history)[1:-1]:
            dist += branch.track.length

        return 1.0 - (self.length - dist) / tail.track.length

    @tail_progress.setter
    def reverse_progress(self, _: float) -> float:
        raise AttributeError("Train.reverse_progress is read-only")

    def step(self, dt: float):
        distance = dt * self.speed

        while distance > 0.0:
            from_ = self.history[0]
            to = from_.to()

            assert from_.track is to.track, (
                "From and to receptors must point to the same track"
            )

            track = from_.track
            distance_to_switch = (1.0 - self.head_progress) * track.length

            if distance < distance_to_switch:
                self.head_progress += distance / track.length
                distance = 0.0
                continue

            else:
                distance -= distance_to_switch
                next_branch = to.pass_through()
                self.history.appendleft(next_branch)
                self.head_progress = 0.0
                continue

        self.trim()
