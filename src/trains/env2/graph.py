from trains.env2.components import Switch, Train


class Graph:
    switches: list[Switch]
    trains: list[Train]

    def _embed_switch(self, switch: Switch) -> list[float]:
        return []  # TODO: Stub

    def _embed_connection(self, connection: Switch.Connection) -> list[float]:
        return []  # TODO: Stub
