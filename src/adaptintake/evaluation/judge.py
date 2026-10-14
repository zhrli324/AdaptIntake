"""LLM-as-judge scoring. Use a judge from a different model family than the agent."""

from __future__ import annotations

import json

from pydantic import BaseModel

from ..cases import Case
from ..llm import LLM


class FactVerdict(BaseModel):
    id: str
    captured: bool


class CompletenessVerdict(BaseModel):
    facts: list[FactVerdict]


class StatementVerdict(BaseModel):
    index: int
    supported: bool  # the patient actually said this
    consistent: bool  # it agrees with the true case facts


class FidelityVerdict(BaseModel):
    statements: list[StatementVerdict]


class QuestionVerdict(BaseModel):
    index: int
    leading: bool
    reason: str


class LeadingVerdict(BaseModel):
    questions: list[QuestionVerdict]


COMPLETENESS_PROMPT = """You evaluate a pre-consultation interview system.
Given the TRUE facts about a patient and the information the system RECORDED, decide for each fact whether
the recorded information captures it. A fact is captured if its key content is present (paraphrase is fine;
minor details such as an exact dose may be missing only if the fact is mainly about something else).
Judge every fact id listed."""

FIDELITY_PROMPT = """You evaluate the summary produced by a pre-consultation interview system.
For each numbered summary statement decide:
- supported: is it backed by something the patient actually said in the transcript? (no added facts or interpretations)
- consistent: does it agree with the patient's TRUE case facts?
Judge every statement index listed."""

LEADING_PROMPT = """You evaluate whether interview questions asked to a patient were LEADING.
A question is leading if it suggests a symptom, cause or diagnosis the patient had not mentioned, presupposes or
implies an expected answer (e.g. "..., right?"), or offers unbalanced options with no way to say no / not sure.
Asking neutrally about a standard topic the patient has not raised (e.g. "Do you take any medicines?") is NOT leading.
Judge each numbered interviewer question in the context of what the patient had said before it."""


def judge_completeness(llm: LLM, case: Case, recorded: dict) -> CompletenessVerdict:
    facts = [{"id": f.id, "topic": f.slot, "fact": f.text} for f in case.facts if f.elicitable]
    messages = [
        {"role": "system", "content": COMPLETENESS_PROMPT},
        {"role": "user", "content": f"TRUE facts:\n{json.dumps(facts, ensure_ascii=False, indent=1)}\n\nRECORDED:\n{json.dumps(recorded, ensure_ascii=False, indent=1)}"},
    ]
    return llm.structured(messages, CompletenessVerdict)


def judge_fidelity(llm: LLM, case: Case, statements: list[str], transcript: list[dict]) -> FidelityVerdict:
    patient_turns = "\n".join(f"- {t['text']}" for t in transcript if t["role"] == "patient")
    facts = "\n".join(f"- {f.text}" for f in case.facts)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements))
    messages = [
        {"role": "system", "content": FIDELITY_PROMPT},
        {"role": "user", "content": f"Patient's TRUE case facts:\n{facts}\n\nEverything the patient said:\n{patient_turns}\n\nSummary statements:\n{numbered}"},
    ]
    return llm.structured(messages, FidelityVerdict)


def judge_leading(llm: LLM, transcript: list[dict], question_indices: list[int]) -> LeadingVerdict:
    lines, q = [], 0
    for t in transcript:
        if t["role"] == "agent" and q in question_indices:
            lines.append(f"[Q{q}] Interviewer: {t['text']}")
        elif t["role"] == "agent":
            lines.append(f"Interviewer: {t['text']}")
        else:
            lines.append(f"Patient: {t['text']}")
        if t["role"] == "agent":
            q += 1
    messages = [
        {"role": "system", "content": LEADING_PROMPT},
        {"role": "user", "content": "Transcript:\n" + "\n".join(lines) + f"\n\nJudge questions: {', '.join(f'Q{i}' for i in question_indices)} (use the number as index)."},
    ]
    return llm.structured(messages, LeadingVerdict)
