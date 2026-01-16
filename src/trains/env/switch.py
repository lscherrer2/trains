from __future__ import annotations

from typing import Literal

from trains.env.branch import Branch


class SwitchPassthroughError(Exception):
    def __init__(self, switch: Switch, from_: Branch):
        self.switch = switch
        self.from_ = from_

    def __str__(self):
        return (
            f"Attempted to pass through switch {self.switch.tag} from "
            f"{self.from_.tag}"
        )


class Switch:
    def __init__(self, tag: str | int, state: bool = False):
        self.tag = tag
        self.approach = Branch(self, tag_suffix="approach")
        self.through = Branch(self, tag_suffix="through")
        self.diverge = Branch(self, tag_suffix="diverging")
        self.diverging = self.diverge
        self.state = state

    def __str__(self) -> str:
        return str(self.tag)

    @property
    def branches(self) -> set[Branch]:
        return {self.approach, self.through, self.diverge}

    def get_branch(self, branch: Literal["approach", "through", "diverge"]):
        return {
            "approach": self.approach,
            "through": self.through,
            "diverge": self.diverge,
            "diverging": self.diverge,
        }[branch]

    def pass_through(self, from_: Branch) -> Branch:
        if from_ is self.approach:
            return self.through if not self.state else self.diverge

        # Coming from through or diverging - check if switch is in correct state
        wrong_state = (
            self.state
            and from_ is self.through
            or not self.state
            and from_ is self.diverge
        )
        if wrong_state:
            raise SwitchPassthroughError(self, from_)

        # If state is correct, return to approach
        return self.approach
