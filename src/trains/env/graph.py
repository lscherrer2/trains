from __future__ import annotations
from trains.env.components import Switch, TrackSegment
import networkx as nx
from enum import Enum


class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2


class Track:
    switches: list[Switch]
    tracks: list[TrackSegment]

    def to_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()

        forward_switches: dict[int, Switch] = {
            2 * i: switch for i, switch in enumerate(self.switches)
        }
        forward_ids: dict[Switch, int] = {
            switch: i for i, switch in forward_switches.items()
        }
        backward_switches: dict[int, Switch] = {
            2 * i + 1: switch for i, switch in enumerate(self.switches)
        }
        backward_ids: dict[Switch, int] = {
            switch: i for i, switch in backward_switches.items()
        }

        # Add nodes for each forward switch
        for id_, switch in forward_switches.items():
            G.add_node(id_, x=None)  # Replace with PyG data

        # Add edges between forward switches
        for id_, switch in forward_switches.items():
            through_id = forward_ids[switch.through.other(switch)]
            diverging_id = forward_ids[switch.diverging.other(switch)]
            G.add_edge(through_id, diverging_id, x=None)  # Replace with PyG data

        # Add nodes for each backward switch
        for id_, switch in backward_switches.items():
            G.add_node(id_, x=None)  # Replace with PyG data

        # Add edges between backward switches
        for id_, switch in backward_switches.items():
            through_id = backward_ids[switch.through.other(switch)]
            diverging_id = backward_ids[switch.diverging.other(switch)]
            G.add_edge(through_id, diverging_id, x=None)  # Replace with PyG data

        return nx.DiGraph()  # STUB
