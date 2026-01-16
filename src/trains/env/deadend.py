from __future__ import annotations

from trains.env.base import Node
from trains.env.branch import Branch


class DeadEndCollision(Exception):
    def __init__(self, dead_end: DeadEnd):
        self.dead_end = dead_end

    def __str__(self) -> str:
        return (
            f"Attempted to pass through dead end with tag {self.dead_end.tag}"
        )


class DeadEnd(Node):
    def __init__(self, tag: str | int = "deadend"):
        self.tag = tag
        self.branch = Branch(self, "branch")

    @property
    def branches(self) -> set[Branch]:
        return {self.branch}

    def pass_through(self, from_: Branch) -> Branch:
        raise DeadEndCollision(self)
