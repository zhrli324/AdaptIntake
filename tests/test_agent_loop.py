"""End-to-end loop with a scripted model: no network calls."""

import json

from adaptintake.agent import FixedQuestionnaire, IntakeAgent
from adaptintake.cases import load_cases
from adaptintake.evaluation.metrics import deterministic_metrics
from adaptintake.llm import ScriptedLLM
from adaptintake.redflags import screen
from adaptintake.schema import SLOT_ORDER

# Scripted patient: each reply answers one slot, with an exact quote.
REPLIES = {
    "chief_complaint": "I get dizzy when I stand up",
    "onset": "it started about ten days ago",
    "provocation": "it is worse in the morning",
    "quality": "it feels swimmy",
    "region": "it is in my head",
    "severity": "about six out of ten",
    "timing": "it comes and goes",
    "associated_symptoms": "nothing else really",
    "past_history": "high blood pressure for years",
    "medications": "I can't remember the name",
    "allergies": "no allergies",
    "functional_change": "I stopped going to my lunch club",
    "main_concern": "I worry it is the new tablet",
}
REPLY_TO_SLOT = {v: k for k, v in REPLIES.items()}


def responder(messages, json_output):
    system = messages[0]["content"]
    if system.startswith("You extract"):
        reply = messages[-1]["content"].split("Patient's latest reply: ")[-1]
        slot = REPLY_TO_SLOT.get(reply)
        updates = []
        if slot:
            kind = "unknown" if slot == "medications" else "answer"
            updates = [{"slot": slot, "value": reply, "evidence": reply, "kind": kind, "vague": False, "contradicts_previous": False}]
        return json.dumps({"updates": updates, "off_topic": False, "wants_to_stop": False})
    if system.startswith("You check"):
        return json.dumps({"leading": False, "reason": "neutral"})
    if system.startswith("You write"):
        request = next(m["content"] for m in messages if "Topic(s) to ask about: " in m["content"])
        topics = request.split("Topic(s) to ask about: ")[1].split("\n")[0]
        question = f"Could you tell me about {topics.split(' (')[0]}?"
        if json_output:  # closed question: text plus answer options
            return json.dumps({"question": question, "options": ["Yes", "No"]})
        return question
    raise AssertionError(system[:40])


def next_reply(agent):
    slot = agent.questions[-1]["slots"][0]
    return REPLIES[slot]


def run(agent):
    turn = agent.start()
    for _ in range(40):
        turn = agent.respond(next_reply(agent))
        if turn.done:
            break
    return agent


def test_adaptive_agent_completes_and_summarises():
    agent = run(IntakeAgent(ScriptedLLM(responder), adaptive=True))
    assert agent.done and agent.end_reason
    assert not agent.errors
    summary = agent.summary()
    assert "chief_complaint" in [i.slot for i in summary.told]
    assert "medications" in [i.slot for i in summary.unclear]  # unknown -> still unclear, never re-asked
    asked = [s for q in agent.questions for s in q["slots"]]
    assert asked.count("medications") == 1
    assert agent.questions[-1]["guidance"] in ("confirm", "focused", "scaffolded")


def test_fixed_questionnaire_asks_every_topic_in_order():
    agent = run(FixedQuestionnaire(ScriptedLLM(responder)))
    asked = [q["slots"][0] for q in agent.questions]
    assert asked == SLOT_ORDER
    assert agent.end_reason == "questionnaire complete"


def test_red_flag_stops_interview():
    agent = IntakeAgent(ScriptedLLM(responder))
    agent.start()
    turn = agent.respond("I have a crushing pressure in my chest going down my left arm")
    assert turn.done and agent.red_flag == "chest pain or pressure"


def test_cases_are_valid_and_red_flags_detectable():
    cases = load_cases()
    assert len(cases) == 20
    for case in cases:
        ids = [f.id for f in case.facts]
        assert ids == [f"F{i}" for i in range(1, len(ids) + 1)]
        if case.red_flag:
            assert screen(case.fact(case.red_flag.fact_id).text)
        else:
            assert all(screen(f.text) is None for f in case.facts)


def test_deterministic_metrics():
    agent = run(FixedQuestionnaire(ScriptedLLM(responder)))
    convo = agent.export()
    case = load_cases(ids=["case_01"])[0]
    m = deterministic_metrics(case, convo)
    assert m["questions"] == len(SLOT_ORDER)
    assert m["unnecessary_questions"] == 0
