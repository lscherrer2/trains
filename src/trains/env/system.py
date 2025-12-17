from __future__ import annotations
from trains.env.components import Switch, Branch, BranchType
from trains.env.entities import Train
from trains.exceptions import TrainOverlapError
from typing import overload
from itertools import islice
from networkx import DiGraph
from bidict import bidict
from math import floor
from numpy.typing import NDArray
import numpy as np


class System:
    switches: list[Switch]
    trains: list[Train]

    def __init__(self, switches: list[Switch], trains: list[Train]):
        self.switches = switches
        self.trains = trains

    def step(self, dt: float) -> System:
        for train in self.trains:
            train.step(dt)
        return self

    @overload
    def set_switch_state(self, switch: Switch, state: bool): ...

    @overload
    def set_switch_state(self, switch: str | int, state: bool): ...

    def set_switch_state(self, switch: Switch | str | int, state: bool):
        """Update the state of a switch to switched, not switched

        Args:
            switch (Switch | str | int): The switch to update state for
            state (bool): True maps the switch to diverging, False to through.

        Raises:
            TrainOverlapError: If a train overlaps the switch it will raise this
            exception.
            ValueError: If a train overlaps the switch it will raise this
            exception.
        """
        if isinstance(switch, (str, int)):
            switch = self.switch_map[switch]

        # Check that no train overlaps switch
        if self.is_switch_overlapped(switch):
            raise TrainOverlapError("Cannot flip switch while train overlaps")

        switch.state = state

    def is_switch_overlapped(self, switch: Switch | str | int) -> bool:
        if isinstance(switch, (str, int)):
            switch = self.switch_map[switch]

        for train in self.trains:
            train.trim()
            for b in islice(train.history, 1, None):
                if b.parent is switch:
                    return True

        return False

    @property
    def switch_map(self) -> dict[int | str, Switch]:
        return {s.tag: s for s in self.switches if s.tag}

    @switch_map.setter
    def switch_map(self, _):
        raise AttributeError("`switch_map` is read-only")

    @property
    def train_map(self) -> dict[int | str, Train]:
        return {t.tag: t for t in self.trains if t.tag}

    @train_map.setter
    def train_map(self, _):
        raise AttributeError("`train_map` is read-only")

    @classmethod
    def from_json(cls, json: dict) -> System:
        from trains.serialization.util import system_from_json

        return system_from_json(json)

    def encode(self, edge_subdivisions: int = 10) -> DiGraph:
        G = DiGraph()

        f_switch_map = bidict()
        b_switch_map = bidict()

        for i, switch in enumerate(self.switches):
            f_switch_map |= {switch: 2 * i}
            b_switch_map |= {switch: 2 * i + 1}

        # Encode forward and backward switches
        for i, switch in enumerate(self.switches):
            # Orthogonal vector encoding unique to each switch
            orth = np.zeros((len(self.switches),), dtype=np.float32)
            orth[i] = 1.0

            # Concatenate with switch encoding state
            data = np.concatenate((switch.encode(), orth))

            # Add nodes for forward and backward switch
            G.add_node(f_switch_map[switch], x=data)
            G.add_node(b_switch_map[switch], x=data)

        # Encode and add edges
        for switch in self.switches:
            for branch in (switch.approach, switch.through, switch.diverging):
                from_ = branch
                to = branch.to()

                from_node = (
                    b_switch_map[from_.parent]
                    if from_.type_ is BranchType.APPROACH
                    else f_switch_map[from_.parent]
                )
                to_node = (
                    f_switch_map[to.parent]
                    if to.type_ is BranchType.APPROACH
                    else b_switch_map[to.parent]
                )

                edge_data = np.concatenate(
                    (branch.encode(), self.encode_overlap(branch, edge_subdivisions))
                )
                G.add_edge(from_node, to_node, x=edge_data)

        return G

    def encode_overlap(self, branch: Branch, segments: int) -> NDArray[np.float32]:  # type: ignore
        overlap = np.zeros((segments,), dtype=np.float32)
        for train in self.trains:
            train.trim()

            hist = train.history
            head = hist[0]
            tail = hist[-1]

            hs = int(floor(segments * float(np.clip(train.head_progress, 0.0, 1.0))))
            ts = int(floor(segments * float(np.clip(train.tail_progress, 0.0, 1.0))))

            if len(hist) >= 3 and any(
                b is branch for b in islice(hist, 1, len(hist) - 1)
            ):
                overlap[:] = 1.0
                break

            if len(hist) == 1 and branch is head and branch is tail:
                start = min(ts, hs)
                end = max(ts, hs)
                overlap[start:end] = 1.0
                continue

            if branch is head:
                overlap[:hs] = 1.0

            if branch is tail:
                overlap[ts:] = 1.0

        return overlap
