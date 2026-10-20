"""AdaptIntake demo.

    streamlit run app/streamlit_app.py

Three modes:
  Talk         you are the patient (type, or tap an answer button for closed questions)
  Simulate     watch the agent interview a simulated patient from a case card, turn by turn
  Replay       step through a saved conversation without any API calls; optionally two side by side
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from adaptintake.agent import SYSTEM_LABELS, SYSTEMS
from adaptintake.cases import CASE_SET_NAMES, load_cases
from adaptintake.config import RESULTS_DIR, model_settings
from adaptintake.llm import LLMError, get_llm
from adaptintake.schema import SLOTS
from adaptintake.simulator import PERSONAS, SimulatedPatient
from adaptintake.summary import build_summary
from adaptintake.schema import InformationState

DEMO_DIR = RESULTS_DIR / "demos"
STATUS_ICON = {"reported": "✅", "missing": "⬜", "conflicting": "⚠️", "unknown": "❔", "declined": "🚫"}

st.set_page_config(page_title="AdaptIntake", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="st-"] { font-size: 19px; }
      [data-testid="stChatMessage"] p { font-size: 1.25rem; line-height: 1.6; }
      [data-testid="stChatInput"] textarea { font-size: 1.2rem; }
      .stButton button { font-size: 1.1rem; padding: 0.45rem 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- shared views -------------------------------------------------------------------------------


def show_transcript(transcript: list[dict]) -> None:
    for t in transcript:
        with st.chat_message("assistant" if t["role"] == "agent" else "user"):
            st.markdown(t["text"])


def show_state(snapshot: dict) -> None:
    rows = [
        {
            "": STATUS_ICON.get(v["status"], ""),
            "topic": SLOTS[name].label,
            "status": v["status"] + (" (vague)" if v.get("vague") else ""),
            "what the patient said": v.get("value") or "",
        }
        for name, v in snapshot.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def show_decision(record: dict | None, title: str = "Why this question") -> None:
    if not record:
        return
    st.markdown(f"**{title}**")
    cols = st.columns(3)
    cols[0].metric("Guidance level", record.get("guidance") or "-", help="Set from how clearly the patient is answering, never from age.")
    cols[1].metric("Question form", record.get("stage") or "-")
    cols[2].metric("Topic", ", ".join(SLOTS[s].label for s in record.get("slots", [])))
    st.caption(record.get("reason", ""))
    scores = record.get("scores") or {}
    if scores:
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        st.caption("Top gap priorities: " + " · ".join(f"{SLOTS[k].label} {v:.2f}" for k, v in top))
    if record.get("rejected_drafts"):
        with st.expander(f"{len(record['rejected_drafts'])} draft(s) rejected by the neutrality check"):
            st.json(record["rejected_drafts"])


def show_summary(summary_md: str, red_flag: str | None) -> None:
    st.subheader("Summary for the patient to check")
    if red_flag:
        st.error("Please seek urgent medical help now. Tell a member of staff or call your local emergency number.")
    st.markdown(summary_md, unsafe_allow_html=True)


def agent_panel(agent) -> None:
    st.subheader("Agent view")
    show_decision(agent.questions[-1] if agent.questions else None, "Latest question")
    show_state(agent.state.snapshot())
    if getattr(agent, "red_flag_overrides", None):
        st.info("Red-flag keyword vetoed as a false alarm: " + "; ".join(o["reason"] for o in agent.red_flag_overrides))
    if agent.red_flag:
        st.warning(f"Red flag: {agent.red_flag}")
    resolved = sum(agent.state[s].is_resolved for s in SLOTS)
    st.caption(f"{len(agent.questions)} questions asked · {resolved}/{len(SLOTS)} topics resolved")


def save_recording(agent, extra: dict) -> Path:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = DEMO_DIR / f"{agent.name}__{extra.get('case', 'live')}__{extra.get('persona', 'you')}__{stamp}.json"
    path.write_text(json.dumps({**agent.export(), **extra}, ensure_ascii=False, indent=1))
    return path


def make_llm(role: str):
    try:
        return get_llm(role)
    except LLMError as err:
        st.error(f"Cannot reach the model ({err}). Check your .env file, or use Replay mode.")
        st.stop()


# ---- sidebar ------------------------------------------------------------------------------------

with st.sidebar:
    st.header("AdaptIntake")
    mode = st.radio("Mode", ["Talk", "Simulate", "Replay"], captions=["You are the patient", "Watch a simulated patient", "Saved conversations, no API"])
    show_reasoning = st.toggle("Show how the agent decides", value=True)
    st.caption(f"Model: {model_settings('agent').model}")
    st.caption("Course prototype. Not a medical device; do not enter real patient data.")


# ---- Talk ---------------------------------------------------------------------------------------

def talk_mode() -> None:
    system = st.sidebar.selectbox("System", list(SYSTEMS), index=list(SYSTEMS).index("adaptintake"), format_func=SYSTEM_LABELS.get)
    if st.sidebar.button("New conversation") or st.session_state.get("talk_system") != system:
        agent = SYSTEMS[system](make_llm("agent"))
        st.session_state.update(talk_agent=agent, talk_system=system, talk_confirmed=None)
        agent.start()
    agent = st.session_state.talk_agent

    def send(text: str) -> None:
        with st.spinner("Thinking..."):
            agent.respond(text)

    chat, panel = st.columns([3, 2]) if show_reasoning else (st.container(), None)
    with chat:
        st.title("Getting ready for your appointment")
        show_transcript(agent.transcript)
        if not agent.done:
            options = agent.questions[-1].get("options") if agent.questions else None
            if options:
                cols = st.columns(len(options))
                for col, option in zip(cols, options):
                    if col.button(option, key=f"opt-{len(agent.questions)}-{option}", width="stretch"):
                        send(option)
                        st.rerun()
            reply = st.chat_input("Type your answer here")
            if reply:
                send(reply)
                st.rerun()
        else:
            show_summary(agent.summary().to_markdown(), agent.red_flag)
            if st.session_state.talk_confirmed is None:
                corrections = st.text_area("Is anything wrong or missing? Write it here (optional).")
                if st.button("This is correct, share it with my doctor", type="primary"):
                    st.session_state.talk_confirmed = save_recording(agent, {"patient_corrections": corrections or None})
                    st.rerun()
            else:
                st.success(f"Thank you. Your summary has been confirmed (saved to {st.session_state.talk_confirmed.name}).")
    if panel is not None:
        with panel:
            agent_panel(agent)


# ---- Simulate -----------------------------------------------------------------------------------

def simulate_mode() -> None:
    case_set = st.sidebar.selectbox("Case set", CASE_SET_NAMES)
    older_only = st.sidebar.toggle("Only patients aged 65+", value=True)
    cases = {c.id: c for c in load_cases(case_set, min_age=65 if older_only else None)}
    if not cases:
        st.info("No cases in this set match the filter.")
        return
    case_id = st.sidebar.selectbox("Case", list(cases), format_func=lambda i: f"{i} · {cases[i].title}")
    persona = st.sidebar.selectbox("Communication style", list(PERSONAS))
    system = st.sidebar.selectbox("System", list(SYSTEMS), index=list(SYSTEMS).index("adaptintake"), format_func=SYSTEM_LABELS.get)
    key = (case_set, case_id, persona, system)
    if st.sidebar.button("Start", type="primary") or st.session_state.get("sim_key") != key:
        agent = SYSTEMS[system](make_llm("agent"))
        patient = SimulatedPatient(make_llm("patient"), cases[case_id], persona)
        agent.start()
        st.session_state.update(sim_key=key, sim_agent=agent, sim_patient=patient, sim_saved=None)
    agent, patient, case = st.session_state.sim_agent, st.session_state.sim_patient, cases[case_id]

    def step() -> None:
        agent.respond(patient.reply(agent.transcript[-1]["text"]))

    st.title(f"{case.title}")
    st.caption(f"Simulated {case.profile.age}-year-old {case.profile.sex} · style: {persona} · system: {SYSTEM_LABELS[system]}")
    with st.expander("Hidden case card (what the simulated patient knows)"):
        st.dataframe(pd.DataFrame([{"topic": SLOTS[f.slot].label, "fact": f.text, "known": f.known, "willing": f.willing} for f in case.facts]), hide_index=True, width="stretch")

    b1, b2, _ = st.columns([1, 1, 3])
    if not agent.done:
        if b1.button("Next turn"):
            with st.spinner("Patient answers, agent decides..."):
                step()
            st.rerun()
        if b2.button("Run to the end"):
            with st.status("Running the interview...") as status:
                while not agent.done and agent.turn < 30:
                    step()
                    status.update(label=f"Turn {agent.turn}: {agent.transcript[-1]['text'][:80]}")
            st.rerun()

    chat, panel = st.columns([3, 2]) if show_reasoning else (st.container(), None)
    with chat:
        show_transcript(agent.transcript)
        if agent.done:
            show_summary(agent.summary().to_markdown(), agent.red_flag)
            if st.session_state.sim_saved is None:
                st.session_state.sim_saved = save_recording(agent, {"case": case.id, "persona": persona})
            st.success(f"Saved for replay: {st.session_state.sim_saved.name}")
    if panel is not None:
        with panel:
            agent_panel(agent)


# ---- Replay -------------------------------------------------------------------------------------

def find_recordings() -> list[Path]:
    return sorted(RESULTS_DIR.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if RESULTS_DIR.exists() else []


def replay_column(convo: dict, step: int) -> None:
    """Show the conversation up to agent question `step` (0-based), with the decision behind it."""
    shown, questions_seen = [], -1
    for t in convo["transcript"]:
        if t["role"] == "agent":
            questions_seen += 1
            if questions_seen > step:
                break
        shown.append(t)
    finished = step >= len(convo["questions"])
    st.markdown(f"**{SYSTEM_LABELS.get(convo['system'], convo['system'])}** · {convo.get('case', '')} · {convo.get('persona', '')}")
    show_transcript(convo["transcript"] if finished else shown)
    if finished:
        summary = build_summary(InformationState.model_validate(convo["state"]), convo.get("red_flag"))
        show_summary(summary.to_markdown(), convo.get("red_flag"))
        st.caption(f"Ended: {convo.get('end_reason')} · {len(convo['questions'])} questions")
    elif show_reasoning:
        record = convo["questions"][step]
        show_decision(record)
        if record.get("state"):
            show_state(record["state"])


def replay_mode() -> None:
    files = find_recordings()
    files = [f for f in files if f.name not in ("config.json",)]
    if not files:
        st.info("No saved conversations yet. Run Simulate mode (it saves automatically) or an evaluation run.")
        return
    label = lambda p: str(p.relative_to(RESULTS_DIR))
    first = st.sidebar.selectbox("Conversation", files, format_func=label)
    compare = st.sidebar.toggle("Compare with a second conversation")
    second = st.sidebar.selectbox("Second conversation", files, index=min(1, len(files) - 1), format_func=label) if compare else None
    convos = [json.loads(first.read_text())] + ([json.loads(second.read_text())] if second else [])
    convos = [c for c in convos if "transcript" in c and "questions" in c]
    longest = max(len(c["questions"]) for c in convos)
    st.title("Replay")
    step = st.slider("Step (agent question)", 0, longest, 0, help="Move right to advance the conversation; the last step shows the summary.")
    cols = st.columns(len(convos))
    for col, convo in zip(cols, convos):
        with col:
            replay_column(convo, step)


{"Talk": talk_mode, "Simulate": simulate_mode, "Replay": replay_mode}[mode]()
