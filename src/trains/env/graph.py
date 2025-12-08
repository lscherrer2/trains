from __future__ import annotations
from trains.env.components import Switch, Track, Train, Branch, BranchType
from networkx import DiGraph
from bidict import bidict
from numpy.typing import NDArray
import numpy as np


class Graph:
    switches: list[Switch]
    trains: list[Train]

    def encode(self) -> DiGraph:
        G = DiGraph()

        f_switch_map = bidict()
        b_switch_map = bidict()

        for i, switch in enumerate(self.switches):
            f_switch_map |= {switch: 2 * i}
            b_switch_map |= {switch: 2 * i + 1}

        # Encode forward switches
        for switch in self.switches:
            G.add_node(f_switch_map[switch], x=switch.encode())

        # Encode backward switches
        for switch in self.switches
            G.add_node(b_switch_map[switch], x=switch.encode())

        # Encode through edges
        for switch in self.switches:

            # node -> through -> node (Forward switch node)
            from_mapping = f_switch_map 
            through_switch = switch.through.to().parent
            encoding = switch.through.encode()

            # Connect to the forward or backward node?
            if switch.through.to().type_ == BranchType.APPROACH:
                to_mapping = f_switch_map
            else:
                to_mapping = b_switch_map

            G.add_edge(
                from_mapping[switch], 
                to_mapping[through_switch], 
                x=np.concat((encoding, self.encode_overlap(switch.through)), axis=0),
            )

        return G


    def encode_overlap(self, branch: Branch) -> NDArray[np.float32]: # type: ignore
        # TODO: Implement
        pass




