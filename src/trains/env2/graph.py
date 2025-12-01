from trains.env2.components import Switch, Train
from itertools import chain
from bidict import bidict
import networkx as nx


class Graph:
    switches: list[Switch]
    trains: list[Train]

    """
    ^ ^
    | |  "Forward" switch
    |/
    ^
    |

    | |
    v v
    |/   "Backward" switch
    |
    v

    Each requires seperate nodes in the graph. Every switch must be 
    divided into two nodes in the final graph.
    """

    def compile(self) -> nx.DiGraph:
        G = nx.DiGraph()

        # TODO

        return G
