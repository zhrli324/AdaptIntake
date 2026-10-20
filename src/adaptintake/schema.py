"""Information state: the slot schema and the per-slot status that drives every decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SlotName = Literal[
    "chief_complaint",
    "onset",
    "provocation",
    "quality",
    "region",
    "severity",
    "timing",
    "associated_symptoms",
    "past_history",
    "medications",
    "allergies",
    "functional_change",
    "main_concern",
]


@dataclass(frozen=True)
class SlotSpec:
    name: str
    label: str  # patient-facing wording of the topic
    description: str  # what the slot should contain (shown to the extraction model)
    importance: float  # w_k: clinical importance in [0, 1]
    burden: float  # base patient effort of answering, in [0, 1]
    essential: bool  # must be resolved before the interview may stop


# OPQRST for the presenting symptom, plus history, medications, allergies,
# change in daily function and the patient's own main concern.
SLOTS: dict[str, SlotSpec] = {
    s.name: s
    for s in [
        SlotSpec("chief_complaint", "the main problem", "The main symptom or problem that brings the patient in, in their own words.", 1.0, 0.2, True),
        SlotSpec("onset", "when it started", "When the problem started and how (suddenly or gradually); any trigger at onset.", 0.9, 0.4, True),
        SlotSpec("provocation", "what makes it better or worse", "Activities, positions, food, medicines or anything else that makes it better or worse.", 0.6, 0.4, False),
        SlotSpec("quality", "what it feels like", "What the symptom feels like (e.g. sharp, dull, burning, spinning).", 0.5, 0.4, False),
        SlotSpec("region", "where it is", "Where the symptom is located and whether it spreads anywhere.", 0.6, 0.3, False),
        SlotSpec("severity", "how bad it is", "How severe it is (e.g. 0-10) or how much it affects the patient.", 0.8, 0.3, True),
        SlotSpec("timing", "how it comes and goes", "Pattern over time: constant or on-and-off, how often, how long episodes last, time of day.", 0.6, 0.4, False),
        SlotSpec("associated_symptoms", "other symptoms", "Other symptoms that came with the problem, including relevant 'no' answers.", 0.7, 0.4, False),
        SlotSpec("past_history", "past health problems", "Long-term conditions, past illnesses, operations or hospital stays.", 0.8, 0.5, True),
        SlotSpec("medications", "medicines", "Regular and recent medicines, including new or changed ones, with dose if known.", 0.9, 0.7, True),
        SlotSpec("allergies", "allergies", "Allergies or bad reactions to medicines, or 'no known allergies'.", 0.8, 0.2, True),
        SlotSpec("functional_change", "changes in daily life", "Changes in daily activities, mobility, sleep, appetite or independence.", 0.5, 0.4, False),
        SlotSpec("main_concern", "what worries you most", "What the patient is most worried about or hopes to get from the visit.", 0.5, 0.3, False),
    ]
}
SLOT_ORDER: list[str] = list(SLOTS)


class SlotStatus(str, Enum):
    MISSING = "missing"
    REPORTED = "reported"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"  # the patient does not know
    DECLINED = "declined"  # the patient prefers not to answer


RESOLVED = {SlotStatus.REPORTED, SlotStatus.UNKNOWN, SlotStatus.DECLINED}


class Evidence(BaseModel):
    turn: int
    quote: str


class SlotState(BaseModel):
    name: str
    status: SlotStatus = SlotStatus.MISSING
    value: str | None = None
    vague: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    conflicting_values: list[str] = Field(default_factory=list)
    times_asked: int = 0
    clarified: bool = False  # was vague or conflicting at some point, later resolved
    list_open: bool = False  # list slot (e.g. medicines) that may have more items: ask "anything else?"
    anchors_used: list[str] = Field(default_factory=list)  # recall cues already asked (recall.ANCHORS)

    @property
    def is_resolved(self) -> bool:
        """Resolved = nothing more to ask: a specific answer, unknown, or declined."""
        if self.status == SlotStatus.REPORTED:
            return not self.vague and not self.list_open
        return self.status in (SlotStatus.UNKNOWN, SlotStatus.DECLINED)


class InformationState(BaseModel):
    slots: dict[str, SlotState] = Field(
        default_factory=lambda: {name: SlotState(name=name) for name in SLOT_ORDER}
    )

    def __getitem__(self, name: str) -> SlotState:
        return self.slots[name]

    def open_slots(self) -> list[SlotState]:
        return [s for s in self.slots.values() if not s.is_resolved]

    def essential_open(self) -> list[SlotState]:
        return [s for s in self.open_slots() if SLOTS[s.name].essential]

    def snapshot(self) -> dict:
        """Compact view used in prompts and logs."""
        return {
            name: {
                "status": s.status.value,
                "value": s.value,
                "vague": s.vague,
                **({"conflicting_values": s.conflicting_values} if s.conflicting_values else {}),
            }
            for name, s in self.slots.items()
        }


# ---- LLM output schemas ------------------------------------------------------


class SlotUpdate(BaseModel):
    slot: SlotName
    value: str = Field(description="Concise paraphrase of what the patient said for this slot.")
    evidence: str = Field(description="Exact quote from the patient's latest reply that supports the value.")
    kind: Literal["answer", "unknown", "declined"] = Field(
        description="'answer' if the patient gave information, 'unknown' if they said they do not know or "
        "cannot remember, 'declined' if they prefer not to say."
    )
    vague: bool = Field(
        description="True if the answer is too imprecise to be useful (e.g. 'a while ago', 'not well')."
    )
    contradicts_previous: bool = Field(
        description="True if this value is inconsistent with the value already recorded for the slot."
    )
    no_more: bool = Field(
        default=False,
        description="True if the patient says there is nothing more to add for this topic "
        "(e.g. 'that's all', 'no other tablets', 'nothing else').",
    )


class TurnAnalysis(BaseModel):
    updates: list[SlotUpdate] = Field(default_factory=list)
    off_topic: bool = Field(description="True if the reply is mostly unrelated to the question asked.")
    wants_to_stop: bool = Field(description="True if the patient asks to end the conversation.")


class NeutralityVerdict(BaseModel):
    leading: bool
    reason: str
