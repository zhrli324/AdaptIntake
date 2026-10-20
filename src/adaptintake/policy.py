"""Ask (decision part): rank information gaps, choose the question form, and decide when to stop.

Priority of an open slot k at turn t:

    s_t(k) = w_k * u_t(k) - lambda * b_k(g_t)

w_k   clinical importance (schema.SLOTS)
u_t   remaining uncertainty: conflicting > missing > vague > resolved
b_k   expected patient effort: base burden, scaled by the guidance level, plus a penalty per earlier ask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .clarity import Guidance
from .schema import SLOTS, InformationState, SlotState, SlotStatus


class Stage(str, Enum):
    OPEN = "open"  # "Could you describe ...?"
    FOCUSED = "focused"  # one specific aspect, still free answer
    CLOSED = "closed"  # balanced options including "not sure"
    CLARIFY = "clarify"  # neutral request to reconcile two inconsistent answers


FUNNEL = [Stage.OPEN, Stage.FOCUSED, Stage.CLOSED]

UNCERTAINTY = {"conflicting": 1.5, "missing": 1.0, "list_open": 0.8, "vague": 0.6}
# Scaffolded questions are short and closed, but the patient is struggling: effort weighs more there.
BURDEN_SCALE = {Guidance.CONFIRM: 0.6, Guidance.FOCUSED: 0.8, Guidance.SCAFFOLDED: 1.0}
LIST_SLOTS = {"medications", "past_history", "associated_symptoms"}
FUNNEL_START = {Guidance.CONFIRM: 0, Guidance.FOCUSED: 1, Guidance.SCAFFOLDED: 1}


@dataclass
class PolicyConfig:
    lam: float = 0.5  # weight of patient effort
    repeat_penalty: float = 0.2  # added burden per earlier ask of the same slot
    threshold: float = 0.15  # gaps scoring below this are not worth asking about
    max_asks_per_slot: int = 3
    max_asks_list_slot: int = 5  # list slots may need "anything else?" follow-ups
    max_turns: int = 20
    allow_bundling: bool = True  # confirm level may cover two missing slots in one question
    anchored_lists: bool = False  # list follow-ups use concrete recall cues (recall.py)
    anchored_burden: float = 0.5  # cue-based follow-ups are easy to answer: burden multiplier


@dataclass
class Decision:
    action: str  # "ask" or "stop"
    slots: list[str] = field(default_factory=list)
    stage: Stage | None = None
    guidance: Guidance | None = None
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def uncertainty(slot: SlotState) -> float:
    if slot.status == SlotStatus.CONFLICTING:
        return UNCERTAINTY["conflicting"]
    if slot.status == SlotStatus.MISSING:
        return UNCERTAINTY["missing"]
    if slot.status == SlotStatus.REPORTED and slot.vague:
        return UNCERTAINTY["vague"]
    if slot.status == SlotStatus.REPORTED and slot.list_open:
        return UNCERTAINTY["list_open"]
    return 0.0


def priority(slot: SlotState, guidance: Guidance, cfg: PolicyConfig) -> float:
    spec = SLOTS[slot.name]
    base = spec.burden
    if cfg.anchored_lists and slot.list_open and not slot.vague and slot.status == SlotStatus.REPORTED:
        base *= cfg.anchored_burden
    burden = base * BURDEN_SCALE[guidance] + cfg.repeat_penalty * slot.times_asked
    return spec.importance * uncertainty(slot) - cfg.lam * burden


def stage_for(slot: SlotState, guidance: Guidance) -> Stage:
    if slot.status == SlotStatus.CONFLICTING:
        return Stage.CLARIFY
    if slot.name == "chief_complaint" and slot.times_asked == 0:
        return Stage.OPEN
    if slot.list_open and not slot.vague:
        return Stage.FOCUSED  # "anything else?"
    index = min(FUNNEL_START[guidance] + slot.times_asked, len(FUNNEL) - 1)
    return FUNNEL[index]


def decide(
    state: InformationState,
    guidance: Guidance,
    turns_used: int,
    cfg: PolicyConfig,
) -> Decision:
    candidates = {
        s.name: round(priority(s, guidance, cfg), 3)
        for s in state.open_slots()
        if s.times_asked < (cfg.max_asks_list_slot if s.name in LIST_SLOTS else cfg.max_asks_per_slot)
    }
    if turns_used >= cfg.max_turns:
        return Decision("stop", scores=candidates, guidance=guidance, reason="turn budget reached")
    worth_asking = {k: v for k, v in candidates.items() if v >= cfg.threshold}
    if not worth_asking:
        return Decision("stop", scores=candidates, guidance=guidance, reason="no remaining gap is worth the patient's effort")

    ranked = sorted(worth_asking, key=lambda k: (-worth_asking[k], list(SLOTS).index(k)))
    top = state[ranked[0]]
    stage = stage_for(top, guidance)
    chosen = [top.name]
    if (
        cfg.allow_bundling
        and guidance == Guidance.CONFIRM
        and stage == Stage.OPEN
        and top.name != "chief_complaint"
        and len(ranked) > 1
    ):
        second = state[ranked[1]]
        if second.status == SlotStatus.MISSING and stage_for(second, guidance) == Stage.OPEN:
            chosen.append(second.name)
    reason = (
        f"highest-priority gap: {', '.join(chosen)} "
        f"({top.status.value}{', vague' if top.vague else ''}; score {worth_asking[top.name]})"
    )
    return Decision("ask", chosen, stage, guidance, candidates, reason)
