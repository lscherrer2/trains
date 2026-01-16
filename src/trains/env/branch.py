from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from trains.env.base import Node
    from trains.env.track import Track


class Branch:
    """A branch endpoint of a switch or dead end."""

    def __init__(self, parent: Node, tag_suffix: str):
        self.parent = parent
        self._tag_suffix = tag_suffix
        self.track: Track | None = None

    @property
    def tag(self) -> str | int:
        return str(self.parent.tag) + "_" + self._tag_suffix

    def __str__(self) -> str:
        return str(self.tag)

    def other(self) -> Branch:
        if self.track is None:
            raise RuntimeError("Branch is not connected to a track")
        return self.track.other(self)
