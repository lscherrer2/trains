from __future__ import annotations
from pydantic import BaseModel
from trains.env.graph import Graph
from trains.env.components import Switch, Track, Branch, Train

__all__ = ["graph_from_json"]


class GraphModel(BaseModel):
    switches: list[str | int]
    tracks: list[TrackModel]
    trains: list[TrainModel]


class TrainModel(BaseModel):
    length: float
    speed: float
    head_progress: float
    history: list[TrackModel]


class TrackModel(BaseModel):
    from_: BranchModel
    to: BranchModel
    length: float


class BranchModel(BaseModel):
    switch: str | int
    type: str


def graph_from_json(json: dict) -> Graph:
    model = GraphModel(**json)
    switch_map: dict[str | int, Switch] = {s: Switch() for s in model.switches}

    track_map: dict[tuple[Branch, Branch], Track] = {}
    for track_model in model.tracks:
        a = track_model.from_
        b = track_model.to
        a = switch_map[a.switch].get_branch(a.type)
        b = switch_map[b.switch].get_branch(b.type)
        if (a, b) not in track_map and (b, a) not in track_map:
            track_map |= {(a, b): Track((a, b), track_model.length)}

    trains: list[Train] = []
    for train_model in model.trains:
        history: list[Branch] = []
        for h in train_model.history:
            a = h.from_
            a = switch_map[a.switch].get_branch(a.type)
            history.append(a)

        Train(
            length=train_model.length,
            speed=train_model.speed,
            head_progress=train_model.head_progress,
            span=history,
        )

    return Graph(list(switch_map.values()), trains)
