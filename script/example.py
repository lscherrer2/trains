from __future__ import annotations

import json

from trains.env import System
from trains.env.branch import Branch


def where(train) -> str:
    head_from: Branch = train.history[0]
    head_to = head_from.other()
    a = f"{head_from.parent.tag}"
    b = f"{head_to.parent.tag}"
    return f"{a} -> {b}  p={train.head_progress:0.2f}  v={train.speed:g}"


def main():
    with open("data/example.json") as f:
        system = System.from_json(json.load(f))

    train = system.trains[0]

    steps = 15
    dt = 0.75

    for i in range(steps):
        if i in (5, 12):
            try:
                system.set_switch_state("B", not system.node_map["B"].state)
            except Exception:
                pass

        print(f"step={i:02d}  B={system.node_map['B'].state}  {where(train)}")

        system.step(dt)


if __name__ == "__main__":
    main()
