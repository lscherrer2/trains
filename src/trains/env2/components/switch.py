from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Switch:
    """"""

    state: bool
    approach: Connection
    through: Connection
    diverging: Connection

    class Receptor(Enum):
        APPROACH = 1
        THROUGH = 2
        DIVERGING = 3

    @dataclass
    class Connection:
        start: tuple[Switch, Switch.Receptor]
        end: tuple[Switch, Switch.Receptor]
        distance: float
