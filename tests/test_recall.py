"""Tests for recall support and red-flag verification (no network)."""

import json

from adaptintake.agent import IntakeAgent
from adaptintake.clarity import Guidance
from adaptintake.llm import LLMError, ScriptedLLM
from adaptintake.policy import PolicyConfig, decide
from adaptintake.recall import consolidate, remaining_anchors
from adaptintake.schema import InformationState, SlotStatus


def reported(state, slot, value):
    state[slot].status, state[slot].value = SlotStatus.REPORTED, value


def test_anchors_fill_from_state_and_skip_when_source_missing():
    state = InformationState()
    keys = [k for k, _ in remaining_anchors("medications", state)]
    assert "for_conditions" not in keys  # no conditions mentioned yet
    reported(state, "past_history", "high blood pressure")
    cues = dict(remaining_anchors("medications", state))
    assert "high blood pressure" in cues["for_conditions"]
    state["medications"].anchors_used.append("for_conditions")
    assert "for_conditions" not in dict(remaining_anchors("medications", state))


def test_anchored_follow_up_has_lower_burden():
    state = InformationState()
    for s in state.slots:
        reported(state, s, "x")
    state["medications"].list_open = True
    state["medications"].times_asked = 2
    plain = decide(state, Guidance.SCAFFOLDED, 5, PolicyConfig())
    anchored = decide(state, Guidance.SCAFFOLDED, 5, PolicyConfig(anchored_lists=True))
    assert anchored.scores["medications"] > plain.scores["medications"]


def test_consolidation_adds_only_quoted_information():
    transcript = [
        {"role": "agent", "text": "What brings you in?"},
        {"role": "patient", "text": "Dizzy. Oh and I take aspirin every day."},
    ]
    state = InformationState()
    reported(state, "chief_complaint", "dizzy")
    reply = {"updates": [
        {"slot": "medications", "value": "aspirin daily", "evidence": "I take aspirin every day", "kind": "answer", "vague": False, "contradicts_previous": False},
        {"slot": "allergies", "value": "penicillin", "evidence": "allergic to penicillin", "kind": "answer", "vague": False, "contradicts_previous": False},
    ]}
    llm = ScriptedLLM(lambda m, j: json.dumps(reply))
    added = consolidate(llm, state, transcript)
    assert [u.slot for u in added] == ["medications"]
    assert state["medications"].status == SlotStatus.REPORTED
    assert state["medications"].evidence[0].turn == 1
    assert state["allergies"].status == SlotStatus.MISSING  # quote not in the transcript


def _agent_with(responder):
    agent = IntakeAgent(ScriptedLLM(responder), verify_red_flags=True)
    agent.start()
    return agent


def test_red_flag_verifier_can_veto_a_false_positive():
    def responder(messages, json_output):
        system = messages[0]["content"]
        if system.startswith("A keyword screen"):
            return json.dumps({"emergency": False, "reason": "blocked nose"})
        if system.startswith("You extract"):
            return json.dumps({"updates": [], "off_topic": False, "wants_to_stop": False})
        if system.startswith("You check"):
            return json.dumps({"leading": False, "reason": "ok"})
        return "Could you tell me more?"
    agent = _agent_with(responder)
    turn = agent.respond("My nose is so blocked I can't breathe through it")
    assert not turn.done and agent.red_flag is None
    assert agent.red_flag_overrides[0]["category"] == "severe breathing difficulty"


def test_red_flag_verifier_failure_escalates():
    def responder(messages, json_output):
        if messages[0]["content"].startswith("A keyword screen"):
            raise LLMError("down")
        return json.dumps({"updates": [], "off_topic": False, "wants_to_stop": False})
    agent = _agent_with(responder)
    turn = agent.respond("I have crushing pressure in my chest")
    assert turn.done and agent.red_flag == "chest pain or pressure"
