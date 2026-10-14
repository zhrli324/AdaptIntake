"""AdaptIntake demo: a large-text chat interface with an optional view of the agent's decisions.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from adaptintake.agent import SYSTEMS
from adaptintake.config import model_settings
from adaptintake.llm import LLMError, get_llm
from adaptintake.schema import SLOTS

SYSTEM_NAMES = {"adaptintake": "AdaptIntake (adaptive)", "b2_nonadaptive": "B2: non-adaptive LLM", "b1_fixed": "B1: fixed questionnaire"}

st.set_page_config(page_title="AdaptIntake", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="st-"] { font-size: 20px; }
      [data-testid="stChatMessage"] p { font-size: 1.25rem; line-height: 1.6; }
      [data-testid="stChatInput"] textarea { font-size: 1.2rem; }
      .stButton button { font-size: 1.1rem; padding: 0.5rem 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_session(system: str) -> None:
    try:
        llm = get_llm("agent")
    except LLMError as err:
        st.session_state.error = str(err)
        return
    agent = SYSTEMS[system](llm)
    st.session_state.agent = agent
    st.session_state.system = system
    st.session_state.messages = [{"role": "assistant", "content": agent.start().text}]
    st.session_state.confirmed = None
    st.session_state.error = None


with st.sidebar:
    st.header("Settings")
    system = st.selectbox("System", list(SYSTEM_NAMES), format_func=SYSTEM_NAMES.get)
    show_trace = st.toggle("Show how the agent decides", value=True)
    if st.button("Start a new conversation") or "agent" not in st.session_state or st.session_state.get("system") != system:
        new_session(system)
    st.caption(f"Model: {model_settings('agent').model}")
    st.caption("Prototype for a course project. Not a medical device; do not enter real patient data.")

st.title("Getting ready for your appointment")
if st.session_state.get("error"):
    st.error(f"Could not start the agent: {st.session_state.error}")
    st.stop()

agent = st.session_state.agent
chat_col, trace_col = st.columns([3, 2]) if show_trace else (st.container(), None)

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not agent.done:
        reply = st.chat_input("Type your answer here")
        if reply:
            st.session_state.messages.append({"role": "user", "content": reply})
            with st.spinner("Thinking..."):
                turn = agent.respond(reply)
            st.session_state.messages.append({"role": "assistant", "content": turn.text})
            st.rerun()
    else:
        summary = agent.summary()
        st.subheader("Your summary")
        if summary.red_flag:
            st.error("Please seek urgent medical help now. Tell a member of staff or call your local emergency number.")
        st.markdown(summary.to_markdown(), unsafe_allow_html=True)
        if st.session_state.confirmed is None:
            corrections = st.text_area("Is anything wrong or missing? Write it here (optional).")
            if st.button("This is correct, share it with my doctor"):
                st.session_state.confirmed = {"summary": summary.to_dict(), "patient_corrections": corrections or None}
                st.rerun()
        else:
            st.success("Thank you. Your summary has been confirmed.")
            st.download_button(
                "Download summary (JSON)",
                json.dumps({**st.session_state.confirmed, "transcript": agent.transcript}, ensure_ascii=False, indent=1),
                file_name="adaptintake_summary.json",
            )

if trace_col is not None:
    with trace_col:
        st.subheader("Agent view")
        clarity = getattr(agent, "clarity", None)
        if clarity is not None:
            level = clarity.level.value if getattr(agent, "adaptive", False) else "focused (fixed)"
            st.metric("Guidance level", level, help="Set from the clarity of recent replies, never from age.")
            if clarity.history:
                st.caption(f"Clarity score: {clarity.score:.2f} (last reply signal {clarity.history[-1]['signal']:.2f})")
        if agent.questions:
            last = agent.questions[-1]
            st.markdown(f"**Last question** targets `{', '.join(last['slots'])}` · stage *{last['stage']}* · {last['reason']}")
            if last["rejected_drafts"]:
                with st.expander(f"{len(last['rejected_drafts'])} rejected draft(s)"):
                    st.json(last["rejected_drafts"])
            if last["scores"]:
                top = sorted(last["scores"].items(), key=lambda kv: -kv[1])[:5]
                st.caption("Top gap priorities: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        rows = [
            {"slot": name, "status": s.status.value + (" (vague)" if s.vague else ""), "value": s.value or "", "asked": s.times_asked}
            for name, s in agent.state.slots.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if agent.red_flag:
            st.warning(f"Red flag: {agent.red_flag}")
        st.caption(f"{len(agent.questions)} questions asked · {len([s for s in SLOTS if agent.state[s].is_resolved])}/{len(SLOTS)} topics resolved")
