from collections import deque
from typing import TYPE_CHECKING

from trains.env.branch import Branch
from trains.env.deadend import DeadEndCollision
from trains.env.switch import SwitchPassthroughError


if TYPE_CHECKING:
    from trains.env.track import Track


class Train:
    def __init__(
        self,
        tag: str | int,
        head_branch: Branch,
        head_distance: float,
        length: float,
        speed: float,
    ):
        self.tag: str | int = tag
        self.head_distance: float = head_distance
        self.length = length
        self.speed = speed
        self._history: deque[Branch] = deque([head_branch])

    def __str__(self) -> str:
        return str(self.tag)

    @property
    def head_branch(self) -> Branch:
        return self._history[0]

    @property
    def history(self) -> deque[Branch]:
        return self._history

    @history.setter
    def history(self, value: deque[Branch]):
        self._history = value

    @property
    def track(self) -> "Track":
        if not self.history:
            raise RuntimeError("Train has no history")
        head_branch = self.history[0]
        if head_branch.track is None:
            raise RuntimeError("Train head branch is not connected to a track")
        return head_branch.track

    @property
    def head_progress(self) -> float:
        return self.head_distance / self.track.length

    @head_progress.setter
    def head_progress(self, value: float):
        self.head_distance = value * self.track.length

    @property
    def tail_distance(self) -> float:
        distance_covered = 0.0

        for i, branch in enumerate(self.history):
            if i == 0:
                segment_length = self.head_distance
            else:
                if branch.track is None:
                    return 0.0
                segment_length = branch.track.length

            if distance_covered + segment_length >= self.length:
                distance_into_segment = self.length - distance_covered
                return distance_into_segment

            distance_covered += segment_length
        return 0.0

    def trim(self):
        distance_left = self.length
        new_history: deque[Branch] = deque()

        head_branch = self.history.popleft()
        new_history.append(head_branch)
        distance_left -= self.head_distance

        while distance_left > 0 and self.history:
            branch = self.history.popleft()
            new_history.append(branch)
            if branch.track is not None:
                distance_left -= branch.track.length

        self.history = new_history
        return self

    def step(self, dt: float):
        step_distance = dt * self.speed
        while step_distance > 0:
            current_track = self.track
            remaining_on_track = current_track.length - self.head_distance

            if step_distance < remaining_on_track:
                self.head_distance += step_distance
                step_distance = 0

            else:
                try:
                    step_distance -= remaining_on_track
                    head_branch = self.history[0]
                    next_branch = head_branch.other()
                    next_head_branch = next_branch.parent.pass_through(
                        next_branch
                    )
                    self.history.appendleft(next_head_branch)
                    self.head_distance = 0.0

                except (DeadEndCollision, SwitchPassthroughError) as e:
                    self.head_distance = current_track.length
                    raise e
