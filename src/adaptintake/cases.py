"""Case cards: the hidden facts a simulated patient knows (ground truth for evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .config import CASES_DIR
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


def load_cases(directory: Path = CASES_DIR, ids: list[str] | None = None) -> list[Case]:
    cases = [load_case(p) for p in sorted(directory.glob("case_*.json"))]
    if ids:
        cases = [c for c in cases if c.id in ids]
    return cases
