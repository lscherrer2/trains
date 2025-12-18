from trains.env.components import Switch, BranchType, Branch
from trains.exceptions import SwitchOverlapError, TrainCollisionError
from trains.env.entities import Train
from trains.env.system import System
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from dataclasses import dataclass
from networkx import DiGraph
from itertools import islice
from bidict import bidict
import numpy as np
import torch
import math


@dataclass
class SystemEncoding:
    pyg_graph: Data
    nx_graph: DiGraph
    f_switch_map: bidict[Switch, int]
    b_switch_map: bidict[Switch, int]


@dataclass
class StepResult:
    obeservation: SystemEncoding
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class Action:
    switch_states: dict[Switch, bool]
    train_speeds: dict[Train, float]


class RLSystemAdapter:
    def __init__(
        self,
        system: System,
        *,
        branch_segments: int,
        collision_value: float,
        switch_overlap_value: float,
        speed_factor: float,
        diverging_factor: float,
        max_steps: int | None = None,
    ):
        self.system = system
        self.segments = branch_segments
        self.collision_val = collision_value
        self.switch_overlap_val = switch_overlap_value
        self.speed_fac = speed_factor
        self.diverging_val = diverging_factor
        self.max_steps = max_steps or float("inf")
        self.steps = 0

    def step(self, action: Action, dt: float) -> StepResult:
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for switch, state in action.switch_states.items():
            if state:
                reward += self.diverging_val

            try:
                self.system.set_switch_state(switch, state)
            except SwitchOverlapError:
                reward += self.switch_overlap_val

        for train, speed in action.train_speeds.items():
            train.speed = speed

        try:
            self.system.step(dt)
            self.steps += 1
            if self.steps >= self.max_steps:
                truncated = True
        except TrainCollisionError as e:
            reward += self.collision_val * len(e.collisions)
            terminated = True

        observation = self.encode()
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _encode_switch(self, switch: Switch) -> torch.torch.tensor:
        num_switches = len(self.system.switches)
        orth = torch.eye(num_switches)[self.system.switches.index(switch)]
        state = torch.tensor([float(switch.state)])
        return torch.cat((state, orth))

    def _encode_branch(
        self, branch: Branch, segments: int | None = None
    ) -> torch.torch.tensor:
        _TAG_MAP = {
            BranchType.APPROACH: -1.0,
            BranchType.THROUGH: 1.0,
            BranchType.DIVERGING: 0.0,
        }
        overlap = self._encode_overlap(branch, segments or self.segments)
        state = torch.tensor([branch.track.length, _TAG_MAP[branch.type_]])
        return torch.cat((state, overlap))

    def _encode_overlap(
        self, branch: Branch, segments: int | None = None
    ) -> torch.torch.tensor:
        segments = segments or self.segments
        overlap = torch.zeros((segments,))
        for train in self.system.trains:
            train.trim()

            hist = train.history
            head = hist[0]
            tail = hist[-1]

            hs = int(
                math.floor(segments * float(np.clip(train.head_progress, 0.0, 1.0)))
            )
            ts = int(
                math.floor(segments * float(np.clip(train.tail_progress, 0.0, 1.0)))
            )

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

    def encode(self) -> SystemEncoding:
        G = DiGraph()

        f_switch_map: bidict[Switch, int] = bidict()
        b_switch_map: bidict[Switch, int] = bidict()

        for i, switch in enumerate(self.system.switches):
            f_switch_map |= {switch: 2 * i}
            b_switch_map |= {switch: 2 * i + 1}

        # Encode forward and backward switches
        for i, switch in enumerate(self.system.switches):
            # Add nodes for forward and backward switch
            G.add_node(f_switch_map[switch], x=self._encode_switch(switch))
            G.add_node(b_switch_map[switch], x=self._encode_switch(switch))

        # Encode and add edges
        for switch in self.system.switches:
            for branch in (switch.approach, switch.through, switch.diverging):
                from_ = branch
                to = branch.other()

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

                edge_data = self._encode_branch(branch, self.segments)
                # Avoid attribute name collision with node feature key `x`
                G.add_edge(from_node, to_node, edge_attr=edge_data)

        graph = from_networkx(G)
        return SystemEncoding(
            pyg_graph=graph,
            nx_graph=G,
            f_switch_map=f_switch_map,
            b_switch_map=b_switch_map,
        )
