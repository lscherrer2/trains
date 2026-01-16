"""Train simulation exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from trains.env.switch import Switch
    from trains.env.track import Track
    from trains.env.train import Train


class SwitchOverlapError(Exception):
    """Raised when attempting to flip a switch while a train is overlapping it."""

    def __init__(self, switch: Switch, trains: list[Train]):
        self.switch = switch
        self.trains = trains

    def __str__(self) -> str:
        train_tags = ", ".join(str(t.tag) for t in self.trains)
        return (
            f"Cannot flip switch {self.switch.tag} while trains are overlapping: "
            f"{train_tags}"
        )


class TrainCollisionError(Exception):
    """Raised when trains collide on a track."""

    def __init__(self, trains: list[tuple[Train, Train, Track]]):
        self.trains = trains

    def __str__(self) -> str:
        collisions = []
        for train_a, train_b, track in self.trains:
            collisions.append(
                f"{train_a.tag} and {train_b.tag} on track {track.tag}"
            )
        return f"Train collision(s) detected: {', '.join(collisions)}"
