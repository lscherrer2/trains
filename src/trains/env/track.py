from __future__ import annotations

from typing import TYPE_CHECKING

from trains.env.base import Tagged


if TYPE_CHECKING:
    from trains.env.branch import Branch


class Track(Tagged):
    def __init__(self, ends: tuple[Branch, Branch], length: float):
        self.tag = "track"
        self.ends = ends
        self.length = length

    def other(self, branch: Branch) -> Branch:
        end = (
            self.ends[0]
            if self.ends[1] is branch
            else (self.ends[1] if self.ends[0] is branch else None)
        )

        if end is None:
            raise RuntimeError("Passed invalid branch to `Track.other`")

        return end
