class TrainOverlapError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg
