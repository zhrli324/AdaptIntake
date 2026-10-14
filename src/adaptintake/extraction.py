"""Understand: turn a free-text patient reply into evidence-backed slot updates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .llm import LLM
from .schema import SLOTS, Evidence, InformationState, SlotStatus, SlotUpdate, TurnAnalysis

# Slots that accumulate several items rather than being overwritten.
MULTI_VALUE_SLOTS = {"medications", "past_history", "associated_symptoms"}

EXTRACTION_PROMPT = """You extract structured information from a patient's reply during a pre-consultation interview.
You do NOT diagnose or interpret. Record only what the patient actually said in their LATEST reply.

Slots:
{slot_list}

Rules:
- Create one update per slot that the latest reply gives information about. Ignore slots it does not address.
- "evidence" must be copied word-for-word from the latest reply (a short span, not the whole reply).
- A clear "no" is information: e.g. "I'm not allergic to anything" -> allergies, value "No known allergies".
- If the patient says they do not know or cannot remember something that was asked, use kind "unknown".
- If the patient says they would rather not talk about something, use kind "declined".
- Mark vague=true for answers too imprecise to be useful ("a while ago", "not great", "some pills").
- Set contradicts_previous=true only if the new value is inconsistent with the value already recorded.
- Never add facts, causes or diagnoses the patient did not state."""


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_supported(quote: str, text: str, threshold: float = 0.85) -> bool:
    """True if `quote` appears in `text` (exactly after normalisation, or as a near-exact span)."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    match = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return match.size / len(q) >= threshold


@dataclass
class TurnUpdate:
    """What changed in the state after one patient reply (input to the Assess module)."""

    accepted: list[SlotUpdate] = field(default_factory=list)
    rejected: list[SlotUpdate] = field(default_factory=list)  # evidence not found in the reply
    newly_resolved: list[str] = field(default_factory=list)
    new_conflicts: list[str] = field(default_factory=list)
    vague: list[str] = field(default_factory=list)
    unknown_or_declined: list[str] = field(default_factory=list)
    off_topic: bool = False
    wants_to_stop: bool = False

    @property
    def non_answer(self) -> bool:
        return not self.accepted and not self.off_topic and not self.wants_to_stop


def analyse_reply(
    llm: LLM, state: InformationState, question: str, reply: str
) -> TurnAnalysis:
    slot_list = "\n".join(f"- {name}: {spec.description}" for name, spec in SLOTS.items())
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT.format(slot_list=slot_list)},
        {
            "role": "user",
            "content": (
                f"Already recorded:\n{json.dumps(state.snapshot(), ensure_ascii=False)}\n\n"
                f"Interviewer's question: {question}\n"
                f"Patient's latest reply: {reply}"
            ),
        },
    ]
    return llm.structured(messages, TurnAnalysis)


def apply_analysis(
    state: InformationState, analysis: TurnAnalysis, reply: str, turn: int
) -> TurnUpdate:
    """Deterministically merge the model's proposals into the state."""
    result = TurnUpdate(off_topic=analysis.off_topic, wants_to_stop=analysis.wants_to_stop)
    for update in analysis.updates:
        if not quote_supported(update.evidence, reply):
            result.rejected.append(update)
            continue
        slot = state[update.slot]
        was_resolved = slot.is_resolved
        slot.evidence.append(Evidence(turn=turn, quote=update.evidence))

        if update.kind in ("unknown", "declined"):
            if slot.status != SlotStatus.REPORTED or slot.vague:
                slot.status = SlotStatus.UNKNOWN if update.kind == "unknown" else SlotStatus.DECLINED
                slot.vague = False
            result.unknown_or_declined.append(update.slot)
        elif slot.status == SlotStatus.REPORTED and update.contradicts_previous:
            slot.status = SlotStatus.CONFLICTING
            slot.conflicting_values = [slot.value or "", update.value]
            result.new_conflicts.append(update.slot)
        else:
            if slot.status == SlotStatus.CONFLICTING or (slot.vague and not update.vague):
                slot.clarified = True
            if (
                update.slot in MULTI_VALUE_SLOTS
                and slot.status == SlotStatus.REPORTED
                and slot.value
                and update.value.lower() not in slot.value.lower()
            ):
                slot.value = f"{slot.value}; {update.value}"
            else:
                slot.value = update.value
            slot.status = SlotStatus.REPORTED
            slot.vague = update.vague
            slot.conflicting_values = []
            if update.vague:
                result.vague.append(update.slot)

        result.accepted.append(update)
        if slot.is_resolved and not was_resolved:
            result.newly_resolved.append(update.slot)
    return result
