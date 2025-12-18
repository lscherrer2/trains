from __future__ import annotations
from trains.util import IdentityHash
from enum import Enum


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

    def pass_through(self, from_: Branch | BranchType):
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
    APPROACH = "approach"
    THROUGH = "through"
    DIVERGING = "diverging"

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

    def __init__(self, parent: Switch, type_: BranchType) -> None:
        self.type_ = type_
        self.parent = parent

    def other(self) -> Branch:
        return self.track.other(self)

    def pass_through(self) -> Branch:
        """
        Returns the branch that corresponds with going through the switch via
        this branch.
        """
        return self.parent.pass_through(self)


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
