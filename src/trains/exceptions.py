from trains.env.components import Track
from trains.env.entities import Train


class SwitchOverlapError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class TrainCollisionError(Exception):
    def __init__(self, trains: set[tuple[Train, Train, Track]]):
        self.trains = trains

    def __str__(self) -> str:
        return f"Collision of {len(self.trains)} trains"
