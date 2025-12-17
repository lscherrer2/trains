from __future__ import annotations
from pydantic import BaseModel


class SystemModel(BaseModel):
    switches: list[str | int]
    tracks: list[TrackModel]
    trains: list[TrainModel]


class TrainModel(BaseModel):
    tag: int | str | None
    length: float
    speed: float
    head_progress: float
    history: list[BranchModel]


class TrackModel(BaseModel):
    from_: BranchModel
    to: BranchModel
    length: float


class BranchModel(BaseModel):
    switch: str | int
    type_: str
