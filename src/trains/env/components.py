from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import overload
from trains.util import IdentityHash
import numpy as np
from numpy.typing import NDArray


class BranchType(Enum):
    APPROACH = 1
    THROUGH = 2
    DIVERGING = 3


@dataclass
class Branch:
    parent: Switch
    track: Track
    type_: BranchType

    def to(self) -> Branch:
        return self.track.other(self)

    def pass_through(self) -> Branch:
        """
        Returns the branch that corresponds with going through the switch via
        this branch.
        """
        return self.parent.pass_through(self)

    def encode(self) -> NDArray[np.float32]:
        # Chosen to encode orthogonality
        TAG_MAP = {
            BranchType.APPROACH: -1.0,
            BranchType.THROUGH: 1.0,
            BranchType.DIVERGING: 0.0,
        }
        return np.array([TAG_MAP[self.type_]], dtype=np.float32)


@dataclass
class Track:
    ends: tuple[Branch, Branch]
    length: float

    def other(self, branch: Branch) -> Branch:
        """Given one end of the connection, get the other."""

        if branch not in self.ends:
            raise ValueError("Invalid argument for Track.other()")

        return self.ends[1] if self.ends[0] is branch else self.ends[0]


@dataclass(eq=False, unsafe_hash=False)
class Switch(IdentityHash):
    approach: Branch
    through: Branch
    diverging: Branch
    state: bool

    def encode(self) -> NDArray[np.float32]:
        return np.array(float(self.state), dtype=np.float32)

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


@dataclass
class Train:
    history: list[Branch]
    progress: float
    speed: float

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
            distance_to_switch = (1.0 - self.progress) * track.length

            # Simple: move forward along track
            if distance < distance_to_switch:
                self.progress -= distance / track.length
                return

            # Travel to next switch, deduct distance, repeat step logic
            else:
                distance -= distance_to_switch  # Deduct distance traveled
                next_branch = to.pass_through()  # Determine where next
                self.history.append(next_branch)  # Add next branch to the history
                self.progress = 0.0  # Zero progress for new track
                continue
