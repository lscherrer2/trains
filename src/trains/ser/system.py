from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


__all__ = [
    "SwitchModel",
    "DeadEndModel",
    "SwitchBranchModel",
    "DeadEndBranchModel",
    "BranchModel",
    "TrackModel",
    "TrainModel",
    "SystemModel",
]


class SwitchModel(BaseModel):
    tag: str
    state: bool


class DeadEndModel(BaseModel):
    tag: str


class SwitchBranchModel(BaseModel):
    node: str
    branch: Literal["through", "approach", "diverge"]


class DeadEndBranchModel(BaseModel):
    node: str


BranchModel = SwitchBranchModel | DeadEndBranchModel


class TrackModel(BaseModel):
    from_: BranchModel
    to: BranchModel
    length: float


class TrainModel(BaseModel):
    tag: str
    speed: float
    length: float
    head_distance: float
    head_branch: BranchModel


class SystemModel(BaseModel):
    switches: list[SwitchModel]
    deadends: list[DeadEndModel]
    tracks: list[TrackModel]
    trains: list[TrainModel]
