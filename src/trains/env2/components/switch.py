from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Branch(Enum):
    APPROACH = 1
    THROUGH = 2
    DIVERGING = 3


@dataclass
class Receptor:
    """A part of a switch that 'receives' a track segment. Useful for tracking
    connections between nodes, particularly during graph compilation."""

    parent: Switch
    branch: Branch
    track: TrackSegment


@dataclass
class Switch:
    state: bool
    approach: Receptor
    through: Receptor
    diverging: Receptor


@dataclass
class TrackSegment:
    length: float
    ends: tuple[Receptor, Receptor]
