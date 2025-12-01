from __future__ import annotations
from trains.env2.components import Switch


class Position:
    """"""

    connection: Switch.Connection
    progress: float


class Train:
    """"""

    head: Position
    tail: Position
    path: list[Switch.Connection]
    speed: float
