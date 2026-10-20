"""Case cards: the hidden facts a simulated patient knows (ground truth for evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .config import CASE_SETS
from .schema import SlotName


class Fact(BaseModel):
    id: str
    slot: SlotName
    text: str
    essential: bool
    known: bool = True  # False: the patient cannot recall it (agent should record "unknown")
    willing: bool = True  # False: the patient prefers not to discuss it (agent should accept "declined")

    @property
    def elicitable(self) -> bool:
        return self.known and self.willing


class Profile(BaseModel):
    age: int
    sex: str
    context: str


class Contradiction(BaseModel):
    slot: SlotName
    first_says: str
    fact_id: str


class RedFlag(BaseModel):
    fact_id: str
    description: str


class Case(BaseModel):
    id: str
    title: str
    source: str
    profile: Profile
    facts: list[Fact]
    contradiction: Contradiction | None = None
    red_flag: RedFlag | None = None

    def fact(self, fact_id: str) -> Fact:
        return next(f for f in self.facts if f.id == fact_id)


def load_case(path: Path) -> Case:
    return Case.model_validate(json.loads(path.read_text()))


CASE_SET_NAMES = [*CASE_SETS, "mts_dialog", "agentclinic"]


def load_cases(case_set: str | Path = "synthetic", ids: list[str] | None = None, min_age: int | None = None) -> list[Case]:
    """Load a case set by name, or from a folder of case_*.json files.

    Names: "synthetic" and "mts_curated" (hand-written folders), "mts_dialog" and "agentclinic"
    (public datasets adapted by rules, see datasets.py). `min_age=65` keeps the older-adult subset.
    """
    if isinstance(case_set, Path):
        cases = [load_case(p) for p in sorted(case_set.glob("case_*.json"))]
    elif case_set in CASE_SETS:
        cases = [load_case(p) for p in sorted(CASE_SETS[case_set].glob("case_*.json"))]
    else:
        from .datasets import ADAPTERS  # local import: datasets builds on Case

        if case_set not in ADAPTERS:
            raise ValueError(f"unknown case set {case_set!r}; choose from {CASE_SET_NAMES}")
        cases = ADAPTERS[case_set]()
    if ids:
        cases = [c for c in cases if c.id in ids]
    if min_age is not None:
        cases = [c for c in cases if c.profile.age >= min_age]
    return cases
