from __future__ import annotations
from trains.util import IdentityHash
from numpy.typing import NDArray
from typing import overload
from enum import Enum
import numpy as np


class BranchType(str, Enum):
    APPROACH = 1
    THROUGH = 2
    DIVERGING = 3

    @classmethod
    def from_str(cls, s: str) -> BranchType:
        return {
            "approach": BranchType.APPROACH,
            "through": BranchType.THROUGH,
            "diverging": BranchType.DIVERGING,
        }[s]


class Branch:
    parent: Switch
    track: Track
    type_: BranchType

    # Vaguely encodes orthogonality
    _TAG_MAP = {
        BranchType.APPROACH: -1.0,
        BranchType.THROUGH: 1.0,
        BranchType.DIVERGING: 0.0,
    }

    def __init__(self, parent: Switch, type_: BranchType) -> None:
        self.type_ = type_
        self.parent = parent

    def to(self) -> Branch:
        return self.track.other(self)

    def pass_through(self) -> Branch:
        """
        Returns the branch that corresponds with going through the switch via
        this branch.
        """
        return self.parent.pass_through(self)

    def encode(self) -> NDArray[np.float32]:
        # Encode branch type and track length
        return np.array([self.track.length, self._TAG_MAP[self.type_]], dtype=np.float32)


class Track:
    ends: tuple[Branch, Branch]
    length: float

    def __init__(self, ends: tuple[Branch, Branch], length: float) -> None:
        self.ends = ends
        self.ends[0].track = self
        self.ends[1].track = self
        self.length = length

    def other(self, branch: Branch) -> Branch:
        """Given one end of the connection, get the other."""

        if branch not in self.ends:
            raise ValueError("Invalid argument for Track.other()")

        return self.ends[1] if self.ends[0] is branch else self.ends[0]


class Switch(IdentityHash):
    approach: Branch
    through: Branch
    diverging: Branch
    state: bool
    tag: int | str | None = None

    def __init__(self, tag: str | int | None):
        self.approach = Branch(self, BranchType.APPROACH)
        self.through = Branch(self, BranchType.THROUGH)
        self.diverging = Branch(self, BranchType.DIVERGING)
        self.state = False
        self.tag = tag

    def get_branch(self, type_: BranchType | str) -> Branch:
        if isinstance(type_, str):
            type_ = BranchType.from_str(type_)
        return {
            BranchType.APPROACH: self.approach,
            BranchType.THROUGH: self.through,
            BranchType.DIVERGING: self.diverging,
        }[type_]

    def encode(self) -> NDArray[np.float32]:
        return np.array([float(self.state)], dtype=np.float32)

    @overload
    def pass_through(self, from_: Branch) -> Branch: ...

    @overload
    def pass_through(self, from_: BranchType) -> Branch: ...

    def pass_through(self, from_: Branch | BranchType) -> Branch:
        """Determines what the next branch is through the switch"""

        assert isinstance(from_, (Branch, BranchType)), (
            "from_ must be of type Branch or BranchType"
        )

        if isinstance(from_, Branch):
            if from_ not in (self.approach, self.through, self.diverging):
                raise ValueError("Invalid arguemnt for Switch.pass_through()")

        if from_ is self.approach or from_ is BranchType.APPROACH:
            if self.state:
                return self.diverging
            else:
                return self.through
        else:
            return self.approach


class Train:
    history: list[Branch]
    head_progress: float
    speed: float
    length: float
    tag: int | str | None

    def __init__(
        self,
        length: float,
        speed: float,
        head_progress: float,
        span: list[Branch],
        tag: int | str | None = None,
    ) -> None:
        self.length = length
        self.speed = speed
        self.head_progress = head_progress
        self.history = span
        self.tag = tag
        self.trim()
    
    @property
    def head_distance(self) -> float:
        """Distance the head is from the switch it is departing"""
        return self.head_progress * self.history[-1].track.length
    
    @property
    def tail_distance(self) -> float:
        """Distance the tail is from the switch it is approaching"""
        return (1.0 - self.tail_progress) * self.history[0].track.length

    def trim(self):
        # TODO: refactor w/ deque
        length_so_far = self.history[-1].track.length * self.head_progress
        new_history: list[Branch] = [self.history[-1]]
        for branch in self.history[:-1][::-1]:
            if length_so_far > self.length:
                break

            new_history.insert(0, branch)
            length_so_far += branch.track.length
        self.history = new_history

    @property
    def tail_progress(self) -> float:
        """The progress along the branch of the tail"""
        self.trim()

        length_so_far = self.head_progress * self.history[-1].track.length
        for branch in self.history[1:-1][::-1]:
            length_so_far += branch.track.length

        return 1.0 - (self.length - length_so_far) / self.history[0].track.length

    @tail_progress.setter
    def reverse_progress(self, _: float) -> float:
        raise AttributeError("Train.reverse_progress is read-only")

    def step(self, dt: float):
        """"""
        distance = dt * self.speed

        while distance > 0.0:
            from_ = self.history[-1]
            to = from_.to()

            assert from_.track is to.track, (
                "From and to receptors must point to the same track"
            )

            track = from_.track
            distance_to_switch = (1.0 - self.head_progress) * track.length

            # Simple: move forward along track
            if distance < distance_to_switch:
                self.head_progress += distance / track.length
                distance = 0.0
                continue

            # Travel to next switch, deduct distance, repeat step logic
            else:
                distance -= distance_to_switch  # Deduct distance traveled
                next_branch = to.pass_through()  # Determine where next
                self.history.append(next_branch)  # Add next branch to the history
                self.head_progress = 0.0  # Zero progress for new track
                continue
    
        self.trim()
