from __future__ import annotations
from trains.util import IdentityHash
from numpy.typing import NDArray
from typing import overload
from enum import Enum
import numpy as np


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
        return np.array(
            [self.track.length, self._TAG_MAP[self.type_]], dtype=np.float32
        )


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

