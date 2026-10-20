"""Recall support: helping patients report everything they know, and keeping everything they said.

Patients on several medicines tend to recall one or two and then answer "no" to a generic
"anything else?"; and patients who answer in fragments often mention facts in passing, which
turn-level extraction can miss. Two mechanisms address this:

1. Anchored recall for list slots (after clinical medication reconciliation): instead of a generic
   "anything else?", the agent works through concrete recall cues, such as medicines for each condition
   the patient mentioned, by time of day, by form, or bought without prescription. Cues come from what
   the patient said or are neutral categories, so they do not suggest new symptoms or diagnoses.
2. Conversation-level consolidation: before stopping, the whole conversation is re-read once to
   recover information stated in passing; every addition still needs a verbatim quote.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from .extraction import MULTI_VALUE_SLOTS, quote_supported
from .llm import LLM
from .schema import SLOTS, Evidence, InformationState, SlotStatus, SlotUpdate

# (key, cue). "{conditions}" / "{medications}" are filled from the state; cues whose source is empty are skipped.
ANCHORS: dict[str, list[tuple[str, str]]] = {
    "medications": [
        ("for_conditions", "medicines for the health conditions they mentioned ({conditions})"),
        ("time_of_day", "medicines taken at different times of day (morning, evening, bedtime)"),
        ("forms", "medicines that are not tablets: inhalers, eye drops, creams, injections or patches"),
        ("over_the_counter", "things bought without a prescription: painkillers, vitamins or herbal remedies"),
    ],
    "past_history": [
        ("regular_care", "health conditions they see a doctor or nurse about regularly"),
        ("from_medicines", "the health conditions their medicines are for ({medications})"),
        ("procedures", "operations or hospital stays in the past"),
    ],
    "associated_symptoms": [
        ("general", "changes in appetite, sleep, weight or energy since this started"),
        ("bodily", "changes in bladder or bowel habits, or in breathing, since this started"),
    ],
}

ANCHOR_TEMPLATES = {
    "for_conditions": "Do you take any medicines for {conditions}?",
    "time_of_day": "Thinking about morning, evening and bedtime, are there any other medicines you take?",
    "forms": "Do you use any inhalers, eye drops, creams, injections or patches?",
    "over_the_counter": "Do you take anything you buy yourself, such as painkillers, vitamins or herbal remedies?",
    "regular_care": "Is there any health condition you see a doctor or nurse about regularly?",
    "from_medicines": "What are your medicines for, if you know?",
    "procedures": "Have you had any operations or stays in hospital?",
    "general": "Since this started, have you noticed any change in your appetite, sleep, weight or energy?",
    "bodily": "Since this started, have you noticed any change in your bladder or bowel habits, or your breathing?",
}


def _fill(text: str, state: InformationState) -> str | None:
    sources = {"conditions": state["past_history"].value, "medications": state["medications"].value}
    for key, value in sources.items():
        if "{" + key + "}" in text:
            if not value:
                return None
            text = text.replace("{" + key + "}", value)
    return text


def remaining_anchors(slot: str, state: InformationState) -> list[tuple[str, str]]:
    used = set(state[slot].anchors_used)
    out = []
    for key, cue in ANCHORS.get(slot, []):
        filled = _fill(cue, state)
        if key not in used and filled:
            out.append((key, filled))
    return out


def anchor_template(key: str, state: InformationState) -> str:
    return _fill(ANCHOR_TEMPLATES[key], state) or ANCHOR_TEMPLATES[key].split(" for {")[0] + "?"


# ---- conversation-level consolidation -------------------------------------------------------------

CONSOLIDATION_PROMPT = """You review a whole pre-consultation interview with a patient.
Find information the patient stated ANYWHERE in the conversation that is missing from the current record,
or recorded less completely than the patient said it (e.g. a medicine mentioned in passing, a symptom
mentioned while answering a different question, a timing detail given later).

Slots:
{slot_list}

Rules:
- Only include information the patient actually stated. Do not repeat what is already recorded.
- "evidence" must be copied word-for-word from ONE patient reply.
- Never add facts, causes or diagnoses the patient did not state.
- If nothing is missing, return an empty list."""


class ConsolidationResult(BaseModel):
    updates: list[SlotUpdate] = Field(default_factory=list)


def consolidate(llm: LLM, state: InformationState, transcript: list[dict]) -> list[SlotUpdate]:
    """Re-read the conversation once; merge verified additions into the state. Returns accepted updates."""
    lines, patient_turns, turn, last_q = [], {}, 0, ""
    for t in transcript:
        if t["role"] == "agent":
            last_q = t["text"]
        else:
            turn += 1
            patient_turns[turn] = t["text"]
            lines.append(f"[turn {turn}] Interviewer: {last_q}\n[turn {turn}] Patient: {t['text']}")
    slot_list = "\n".join(f"- {name}: {spec.description}" for name, spec in SLOTS.items())
    messages = [
        {"role": "system", "content": CONSOLIDATION_PROMPT.format(slot_list=slot_list)},
        {"role": "user", "content": f"Current record:\n{json.dumps(state.snapshot(), ensure_ascii=False)}\n\nConversation:\n" + "\n".join(lines)},
    ]
    result = llm.structured(messages, ConsolidationResult)
    accepted = []
    for update in result.updates:
        source = next((n for n, text in patient_turns.items() if quote_supported(update.evidence, text)), None)
        if source is None:
            continue
        slot = state[update.slot]
        if update.kind != "answer":
            if slot.status == SlotStatus.MISSING:
                slot.status = SlotStatus.UNKNOWN if update.kind == "unknown" else SlotStatus.DECLINED
            else:
                continue
        elif slot.status == SlotStatus.MISSING:
            slot.status, slot.value, slot.vague = SlotStatus.REPORTED, update.value, update.vague
        elif slot.status == SlotStatus.REPORTED and slot.vague and not update.vague:
            slot.value, slot.vague, slot.clarified = update.value, False, True
        elif (
            slot.status == SlotStatus.REPORTED
            and update.slot in MULTI_VALUE_SLOTS
            and slot.value
            and update.value.lower() not in slot.value.lower()
        ):
            slot.value = f"{slot.value}; {update.value}"
        else:
            continue  # nothing new, or would create a conflict: leave the record unchanged
        slot.evidence.append(Evidence(turn=source, quote=update.evidence))
        accepted.append(update)
    return accepted
