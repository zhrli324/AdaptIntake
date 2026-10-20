"""The evaluation pipeline end to end (simulate -> score -> analyse) with scripted models: no network."""

import json
import random
import re

from test_agent_loop import REPLIES, responder as agent_responder

from adaptintake.evaluation import analyze, run
from adaptintake.llm import ScriptedLLM
from adaptintake.schema import SLOTS

LABEL_TO_SLOT = {spec.label: name for name, spec in SLOTS.items()}


def patient(messages, json_output):
    question = messages[-1]["content"].lower()
    for label, name in LABEL_TO_SLOT.items():
        if label in question:
            return REPLIES[name]
    return "I'm not sure."


def judge(messages, json_output):
    system, user = messages[0]["content"], messages[-1]["content"]
    if "TRUE facts about a patient" in system:
        return json.dumps({"facts": [{"id": i, "captured": True} for i in re.findall(r'"id": "(F\d+)"', user)]})
    if "summary produced" in system:
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return json.dumps({"statements": [{"index": i, "supported": True, "consistent": True} for i in range(n)]})
    return json.dumps({"questions": [{"index": int(i), "leading": False, "reason": "-"} for i in re.findall(r"\[Q(\d+)\]", user)]})


def test_run_and_analyse(tmp_path, monkeypatch):
    random.seed(0)
    fakes = {"agent": ScriptedLLM(agent_responder), "patient": ScriptedLLM(patient), "judge": ScriptedLLM(judge)}
    monkeypatch.setattr(run, "get_llm", lambda role: fakes[role])
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(analyze, "RESULTS_DIR", tmp_path)
    run.main(["--run-id", "t", "--cases", "case_01", "case_19", "--personas", "clear", "--workers", "2"])
    files = sorted((tmp_path / "t" / "conversations").glob("*.json"))
    assert len(files) == 6  # 2 cases x 1 persona x 3 systems
    analyze.main(["--run-id", "t"])
    assert (tmp_path / "t" / "summary.md").exists()
    assert (tmp_path / "t" / "figures" / "completeness.png").exists()
