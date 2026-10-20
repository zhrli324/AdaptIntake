"""LLM-simulated patients: one case card played in one of four communication styles.

Personas describe how information is disclosed, not who the patient is; they must not rely on
stereotypes about age.
"""

from __future__ import annotations

from .cases import Case
from .llm import LLM

PERSONAS: dict[str, str] = {
    "clear": (
        "You communicate clearly. Answer what is asked fully and precisely, in 1-3 sentences, with concrete "
        "times, numbers and names when you know them. Answer only what was asked."
    ),
    "fragmented": (
        "You communicate in fragments. HARD RULES: each reply is at most 12 words; give at most ONE fact per "
        "reply, even if the question asks about several things; in about one reply out of three, drift to a "
        "loosely related topic (a family member, a past event) instead of answering. Never list several "
        "medicines or conditions in one reply; mention one and stop."
    ),
    "uncertain": (
        "You are often unsure about details. HARD RULES: the FIRST time you are asked about a time, duration, "
        "frequency, amount, dose, score or the name of something, answer vaguely WITHOUT the specific detail "
        "(e.g. 'a while back', 'not too bad', 'some tablets', 'a few times'). Give the specific detail only if "
        "the interviewer asks about that same thing again, offers concrete options, or gives an everyday "
        "reference point (a weekend, a holiday, a meal). Keep replies to 1-2 short sentences and answer only "
        "what was asked."
    ),
    "contradictory": (
        "You sometimes misremember. The first time the topic of the CONTRADICTION comes up, say the "
        "'first says' version. If later asked about it again or asked which version is right, give the true "
        "fact. Otherwise answer only what was asked, in 1-2 sentences."
    ),
}

SIMULATOR_PROMPT = """You are role-playing a patient in a pre-consultation interview, to test an interviewing system.

Who you are: {age}-year-old {sex}. {context}
Presenting problem: {title}

Facts you know (your hidden case card):
{facts}
{contradiction}
Communication style: {persona}

Rules:
- Stay in character and speak like a real person, in the first person, in plain English.
- Only reveal facts from your case card, and only when they are relevant to the question. Do not volunteer
  your whole story at once.
- Facts marked [DOES NOT KNOW]: you know the general situation but NOT the specific detail (a name, dose,
  date or result, even if it appears in brackets on your card). If asked for it, say you cannot remember.
- Facts marked [PREFERS NOT TO SAY]: if asked, politely say you would rather not talk about it.
- If asked about something not on your case card, say no or that you have not noticed anything.
- Never invent new symptoms, medicines or history. Never use medical diagnoses you were not told about.
- Reply with the patient's words only (no stage directions)."""


def format_facts(case: Case) -> str:
    lines = []
    for f in case.facts:
        tag = "" if f.elicitable else (" [DOES NOT KNOW]" if not f.known else " [PREFERS NOT TO SAY]")
        lines.append(f"- ({f.slot}) {f.text}{tag}")
    return "\n".join(lines)


class SimulatedPatient:
    def __init__(self, llm: LLM, case: Case, persona: str):
        if persona not in PERSONAS:
            raise ValueError(f"unknown persona {persona!r}")
        self.llm = llm
        self.case = case
        self.persona = persona
        contradiction = ""
        if persona == "contradictory" and case.contradiction:
            c = case.contradiction
            contradiction = (
                f"\nCONTRADICTION: about {c.slot}, the first time it comes up you say: \"{c.first_says}\". "
                f"The truth is: {case.fact(c.fact_id).text}\n"
            )
        self.system = SIMULATOR_PROMPT.format(
            age=case.profile.age,
            sex=case.profile.sex if case.profile.sex != "unspecified" else "adult",
            context=case.profile.context,
            title=case.title,
            facts=format_facts(case),
            contradiction=contradiction,
            persona=PERSONAS[persona],
        )
        self.history: list[dict] = []

    def reply(self, agent_text: str) -> str:
        self.history.append({"role": "user", "content": agent_text})
        text = self.llm.chat([{"role": "system", "content": self.system}, *self.history])
        self.history.append({"role": "assistant", "content": text})
        return text
