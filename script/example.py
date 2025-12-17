from __future__ import annotations
from trains.env import System
import json


def load_system(fp: str) -> System:
    with open(fp, "r") as f:
        return System.from_json(json.load(f))


def where(train) -> str:
    head_from = train.history[0]
    head_to = head_from.to()
    a = f"{head_from.parent.tag}.{head_from.type_.name.lower()}"
    b = f"{head_to.parent.tag}.{head_to.type_.name.lower()}"
    return f"{a} -> {b}  p={train.head_progress:0.2f}  v={train.speed:g}"


def main():
    system = load_system("data/example.json")
    train = system.trains[0]

    steps = 15
    dt = 0.75

    for i in range(steps):
        if i in (5, 12) and not system.is_switch_overlapped("B"):
            system.set_switch_state("B", not system.switch_map["B"].state)

        print(f"step={i:02d}  B={system.switch_map['B'].state}  {where(train)}")

        system.step(dt)


if __name__ == "__main__":
    main()
