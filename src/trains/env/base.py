from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from trains.env.branch import Branch


class Tagged(Protocol):
    tag: str

    def __str__(self) -> str:
        return str(self.tag)


class Node(Tagged, Protocol):
    @property
    def branches(self) -> set[Branch]: ...

    def pass_through(self, from_: Branch) -> Branch: ...
