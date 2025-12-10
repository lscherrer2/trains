from __future__ import annotations
from pydantic import BaseModel
from trains.env.system import System
from trains.env.components import Switch, Track, Branch, Train


__all__ = ["system_from_json"]


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


def system_from_json(json: dict) -> System:
    model = SystemModel(**json)
    switch_map: dict[str | int, Switch] = {s: Switch(s) for s in model.switches}

    track_map: dict[tuple[Branch, Branch], Track] = {}
    for track_model in model.tracks:
        a = track_model.from_
        b = track_model.to
        a = switch_map[a.switch].get_branch(a.type_)
        b = switch_map[b.switch].get_branch(b.type_)
        if (a, b) not in track_map and (b, a) not in track_map:
            track_map |= {(a, b): Track((a, b), track_model.length)}

    trains: list[Train] = []
    for train_model in model.trains:
        history: list[Branch] = []
        for h in train_model.history:
            a = switch_map[h.switch].get_branch(h.type_)
            history.append(a)

        t = Train(
            length=train_model.length,
            speed=train_model.speed,
            head_progress=train_model.head_progress,
            span=history,
            tag=train_model.tag,
        )

        t.trim()
        trains.append(t)

    return System(list(switch_map.values()), trains)
