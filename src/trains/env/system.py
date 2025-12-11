from __future__ import annotations
from trains.env.components import Switch, Train, Branch, BranchType
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
        from trains.serialization import system_from_json

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
            data = np.concat((switch.encode(), orth), axis=0)

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

                edge_data = np.concat((branch.encode(), self.encode_overlap(branch, edge_subdivisions)), axis=0)
                G.add_edge(
                    from_node, 
                    to_node, 
                    x = edge_data
                )

        return G

    def encode_overlap(self, branch: Branch, segments: int) -> NDArray[np.float32]:  # type: ignore
        overlap = np.zeros((segments,), dtype=np.float32)
        for train in self.trains:
            train.trim()

            if branch in train.history[1:-1]:
                overlap[:] = 1.0
                break

            if branch is train.history[0]:
                split = segments * train.head_progress
                overlap[: floor(split)] = 1.0

            if branch is train.history[-1]:
                split = segments * train.tail_progress
                overlap[floor(split) :] = 1.0

        return overlap
