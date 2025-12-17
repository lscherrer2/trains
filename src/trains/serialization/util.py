from trains.env.system import System
from trains.env.components import Switch, Track, Branch
from trains.env.entities import Train
from trains.serialization.models import SystemModel


__all__ = ["system_from_json"]


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
