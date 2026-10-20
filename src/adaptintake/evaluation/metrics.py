"""Metrics for one finished conversation: deterministic counts plus judge-based scores."""

from __future__ import annotations

from ..cases import Case
from ..llm import LLM, LLMError
from . import judge


def deterministic_metrics(case: Case, convo: dict) -> dict:
    questions = convo["questions"]
    unnecessary = [q["index"] for q in questions if any(q["already_resolved"].values())]
    patient_words = sum(len(t["text"].split()) for t in convo["transcript"] if t["role"] == "patient")
    out = {
        "questions": len(questions),
        "unnecessary_questions": len(unnecessary),
        "unnecessary_rate": len(unnecessary) / len(questions) if questions else 0.0,
        "patient_words": patient_words,
        "end_reason": convo["end_reason"],
        "rejected_extractions": convo["rejected_extractions"],
        "template_fallbacks": sum(q["source"] == "template" for q in questions) if convo["system"] != "fixed_questionnaire" else None,
        "rule_flagged_drafts": sum(len(q["rejected_drafts"]) for q in questions),
    }
    if case.red_flag:
        out["red_flag_detected"] = bool(convo["red_flag"])
        out["red_flag_turn"] = convo["turns"] if convo["red_flag"] else None
    return out


def recorded_view(state: dict) -> dict:
    """What the system recorded, as shown to the completeness judge."""
    view = {}
    for name, slot in state["slots"].items():
        if slot["status"] == "missing":
            continue
        entry = {"status": slot["status"], "value": slot["value"]}
        if slot["conflicting_values"]:
            entry["conflicting_values"] = slot["conflicting_values"]
        view[name] = entry
    return view


def judge_metrics(judge_llm: LLM, case: Case, convo: dict) -> dict:
    out: dict = {}
    try:
        verdict = judge.judge_completeness(judge_llm, case, recorded_view(convo["state"]))
        captured = {v.id: v.captured for v in verdict.facts}
        elicitable = [f for f in case.facts if f.elicitable]
        essential = [f for f in elicitable if f.essential]
        out["completeness"] = sum(captured.get(f.id, False) for f in elicitable) / len(elicitable)
        out["essential_completeness"] = sum(captured.get(f.id, False) for f in essential) / len(essential) if essential else None
        out["captured_facts"] = captured
    except LLMError as err:
        out["judge_error_completeness"] = str(err)

    statements = [f"{s['slot']}: {s['text']}" for s in convo["summary"]["told"] + convo["summary"]["clarified"]]
    if statements:
        try:
            verdict = judge.judge_fidelity(judge_llm, case, statements, convo["transcript"])
            by_index = {v.index: v for v in verdict.statements}
            out["fidelity"] = sum(by_index[i].supported for i in by_index) / len(statements)
            out["accuracy"] = sum(by_index[i].consistent for i in by_index) / len(statements)
        except LLMError as err:
            out["judge_error_fidelity"] = str(err)

    # The opening question is identical across systems, so only later questions are judged.
    indices = [q["index"] for q in convo["questions"] if q["source"] != "opening"]
    if indices:
        try:
            verdict = judge.judge_leading(judge_llm, convo["transcript"], indices)
            leading = [v.index for v in verdict.questions if v.leading]
            out["leading_questions"] = len(leading)
            out["leading_rate"] = len(leading) / len(indices)
            out["leading_details"] = [{"index": v.index, "reason": v.reason} for v in verdict.questions if v.leading]
        except LLMError as err:
            out["judge_error_leading"] = str(err)
    return out
