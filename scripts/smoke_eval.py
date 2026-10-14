"""Offline smoke test of the evaluation pipeline with scripted (fake) models. Numbers are meaningless."""
import sys, json, random, re, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))
from test_agent_loop import responder as agent_responder, REPLIES
from adaptintake.llm import ScriptedLLM
from adaptintake.schema import SLOTS
from adaptintake.evaluation import run as R
from adaptintake.config import RESULTS_DIR
random.seed(0)
LABEL = {spec.label: name for name, spec in SLOTS.items()}
def patient(messages, json_output):
    q = messages[-1]["content"].lower()
    for label, name in LABEL.items():
        if label in q: return REPLIES[name]
    return random.choice(list(REPLIES.values()))
def judge(messages, json_output):
    sysm, user = messages[0]["content"], messages[-1]["content"]
    if "TRUE facts about a patient" in sysm:
        ids = re.findall(r'"id": "(F\d+)"', user)
        return json.dumps({"facts": [{"id": i, "captured": random.random() < .7} for i in ids]})
    if "summary produced" in sysm:
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return json.dumps({"statements": [{"index": i, "supported": True, "consistent": random.random() < .9} for i in range(n)]})
    qs = re.findall(r"\[Q(\d+)\]", user)
    return json.dumps({"questions": [{"index": int(i), "leading": random.random() < .1, "reason": "x"} for i in qs]})
fakes = {"agent": ScriptedLLM(agent_responder), "patient": ScriptedLLM(patient), "judge": ScriptedLLM(judge)}
R.get_llm = lambda role: fakes[role]
R.model_settings = lambda role: type("S", (), {"model": f"fake-{role}"})()
shutil.rmtree(RESULTS_DIR / "smoke", ignore_errors=True)
R.main(["--run-id", "smoke", "--cases", "case_01", "case_02", "case_19", "--workers", "2"])
from adaptintake.evaluation import analyze
analyze.main(["--run-id", "smoke"])
