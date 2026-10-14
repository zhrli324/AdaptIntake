"""Ask (wording part): phrase the selected gap as one short, neutral question.

The LLM phrases the question; a rule check and an optional LLM critic reject leading wording.
If no acceptable question is produced, a neutral template is used instead, so the conversation
never depends on free generation alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .clarity import Guidance
from .llm import LLM, LLMError
from .policy import Decision, Stage
from .schema import SLOTS, InformationState, NeutralityVerdict

OPENING_QUESTION = "Hello, I'm here to help you get ready for your appointment. Could you tell me what has been bothering you?"

# Neutral templates: used by the fixed-questionnaire baseline and as a fallback.
TEMPLATES: dict[str, dict[Stage, str]] = {
    "chief_complaint": {
        Stage.OPEN: "Could you tell me what has been bothering you?",
        Stage.FOCUSED: "What is the main problem you would like the doctor to look at today?",
        Stage.CLOSED: "Is the main problem pain, a change in how your body works, or something else? You can also say you are not sure.",
    },
    "onset": {
        Stage.OPEN: "Could you tell me how and when this began?",
        Stage.FOCUSED: "When did you first notice it?",
        Stage.CLOSED: "Did it start in the last few days, a few weeks ago, or longer ago? It's fine to say you're not sure.",
    },
    "provocation": {
        Stage.OPEN: "Have you noticed anything that changes how it feels?",
        Stage.FOCUSED: "Is there anything that makes it better or worse?",
        Stage.CLOSED: "Does anything make it better, does anything make it worse, or have you not noticed either way?",
    },
    "quality": {
        Stage.OPEN: "Could you describe what it feels like?",
        Stage.FOCUSED: "How would you describe the feeling in your own words?",
        Stage.CLOSED: "Would you call it sharp, dull, burning, or something else? You can say you're not sure.",
    },
    "region": {
        Stage.OPEN: "Could you tell me where you feel it?",
        Stage.FOCUSED: "Where exactly do you feel it, and does it spread anywhere?",
        Stage.CLOSED: "Is it in one place, does it move or spread, or is it hard to say?",
    },
    "severity": {
        Stage.OPEN: "How much is this affecting you?",
        Stage.FOCUSED: "On a scale from 0 to 10, where 10 is the worst, how bad is it?",
        Stage.CLOSED: "Would you say it is mild, moderate or severe? You can say you're not sure.",
    },
    "timing": {
        Stage.OPEN: "Could you tell me how it comes and goes over the day?",
        Stage.FOCUSED: "Is it there all the time, or does it come and go?",
        Stage.CLOSED: "Is it there all the time, does it come and go, or is it hard to say?",
    },
    "associated_symptoms": {
        Stage.OPEN: "Have you noticed any other changes in how you feel?",
        Stage.FOCUSED: "Has anything else been bothering you along with this?",
        Stage.CLOSED: "Have you had any other symptoms at the same time: yes, no, or not sure?",
    },
    "past_history": {
        Stage.OPEN: "Could you tell me about any long-term health conditions you have?",
        Stage.FOCUSED: "Do you see a doctor regularly for any health condition?",
        Stage.CLOSED: "Have you ever stayed in hospital or had an operation: yes, no, or not sure?",
    },
    "medications": {
        Stage.OPEN: "Could you tell me about the medicines you take?",
        Stage.FOCUSED: "Which medicines do you take regularly, including any new ones?",
        Stage.CLOSED: "Have any of your medicines been started or changed recently: yes, no, or not sure?",
    },
    "allergies": {
        Stage.OPEN: "Have you ever had a reaction to a medicine?",
        Stage.FOCUSED: "Are you allergic to any medicines?",
        Stage.CLOSED: "Do you have any medicine allergies: yes, no, or not sure?",
    },
    "functional_change": {
        Stage.OPEN: "How has this affected your day-to-day life?",
        Stage.FOCUSED: "Has this changed what you can do around the house or outside?",
        Stage.CLOSED: "Compared with before, are daily tasks the same, a bit harder, or much harder?",
    },
    "main_concern": {
        Stage.OPEN: "What would you most like the doctor to know or help with?",
        Stage.FOCUSED: "Is there anything about this that worries you?",
        Stage.CLOSED: "Is there something specific you are worried about: yes, no, or not sure?",
    },
}

# "Anything else?" follow-ups for list slots that may still have items.
LIST_FOLLOWUP = {
    "medications": "Apart from those, do you take any other medicines, including ones you buy yourself?",
    "past_history": "Apart from that, have you had any other health problems, operations or hospital stays?",
    "associated_symptoms": "Apart from that, have you noticed anything else that comes with it?",
}

STAGE_INSTRUCTIONS = {
    Stage.OPEN: "Ask an open question that invites the patient to describe it in their own words.",
    Stage.FOCUSED: "Ask about one specific aspect, still letting the patient answer freely.",
    Stage.CLOSED: "Offer 2-3 balanced options and always include 'or you're not sure' as an option.",
    Stage.CLARIFY: "The patient gave two different answers. Mention both neutrally, without suggesting that either is wrong, and ask which is right.",
}

GUIDANCE_INSTRUCTIONS = {
    Guidance.CONFIRM: "The patient communicates clearly. Be brief and natural; you may cover the listed topics together.",
    Guidance.FOCUSED: "Ask about exactly one thing, in plain words.",
    Guidance.SCAFFOLDED: "The patient finds it hard to answer. Use one very short sentence (at most 15 words), "
    "plain everyday words, and if asking about time, offer an everyday anchor (e.g. 'before or after the weekend').",
}

GENERATION_PROMPT = """You write the next question for a pre-consultation interview with an older patient.
Write ONE question only, in plain, friendly English, without medical jargon.
Never suggest a symptom, a cause or a diagnosis the patient has not mentioned.
Never presuppose an answer (no "you've had ..., right?", no "how bad is the pain" if no pain was mentioned).
Never give medical advice. Output only the question text."""

CRITIC_PROMPT = """You check interview questions for neutrality. A question is LEADING if it:
- suggests a symptom, cause or diagnosis the patient has not mentioned;
- presupposes or implies an expected answer (e.g. tag questions like "..., right?", "Surely ...");
- offers unbalanced options (only one side, or no way to say "no"/"not sure").
Asking neutrally about a topic the patient has not raised (e.g. "Do you take any medicines?") is NOT leading."""

TAG_QUESTION = re.compile(r"(?:,\s*(?:right|correct|isn't it|aren't you|don't you|didn't you|haven't you|wasn't it)\s*\?)|\b(?:surely|obviously|i assume|i'm sure you)\b", re.I)
DIAGNOSIS_TERMS = re.compile(
    r"\b(?:heart attack|stroke|cancer|tumou?r|infection|diabetes|dementia|alzheimer|pneumonia|angina|"
    r"arthritis|depression|uti|kidney failure|heart failure|vertigo|parkinson)\b",
    re.I,
)


def rule_check(question: str, patient_text: str) -> list[str]:
    """Deterministic neutrality rules. Returns the list of violations (empty = passed)."""
    problems = []
    if TAG_QUESTION.search(question):
        problems.append("tag question or presupposition")
    mentioned = {m.lower() for m in DIAGNOSIS_TERMS.findall(patient_text)}
    for term in DIAGNOSIS_TERMS.findall(question):
        if term.lower() not in mentioned:
            problems.append(f"introduces unmentioned diagnosis '{term}'")
    if question.count("?") > 2:
        problems.append("more than one question")
    return problems


@dataclass
class QuestionResult:
    text: str
    source: str  # "llm", "template" or "opening"
    attempts: list[dict] = field(default_factory=list)  # rejected drafts and why


def template_question(decision: Decision, state: InformationState) -> str:
    slot = decision.slots[0]
    if state[slot].list_open and not state[slot].vague and slot in LIST_FOLLOWUP:
        return LIST_FOLLOWUP[slot]
    if decision.stage == Stage.CLARIFY:
        a, b = (state[slot].conflicting_values + ["", ""])[:2]
        return f"I want to make sure I have this right about {SLOTS[slot].label}. I noted '{a}' and also '{b}'. Which is closer?"
    return TEMPLATES[slot][decision.stage]


def generate_question(
    llm: LLM,
    decision: Decision,
    state: InformationState,
    transcript: list[dict],
    use_critic: bool = True,
    max_drafts: int = 2,
) -> QuestionResult:
    patient_text = " ".join(t["text"] for t in transcript if t["role"] == "patient")
    recent = "\n".join(f"{t['role']}: {t['text']}" for t in transcript[-6:])
    topics = "; ".join(f"{SLOTS[s].label} ({SLOTS[s].description})" for s in decision.slots)
    extra = ""
    first = state[decision.slots[0]]
    if first.list_open and not first.vague:
        extra = f"\nThe patient already mentioned: {first.value}. Ask only whether there is anything else to add."
    if decision.stage == Stage.CLARIFY:
        values = state[decision.slots[0]].conflicting_values
        extra = f"\nThe two answers were: {values}"
    main_problem = state["chief_complaint"].value or "not yet known"
    request = (
        f"Recent conversation:\n{recent}\n\n"
        f"Main problem (symptom questions refer to this): {main_problem}\n"
        f"Topic(s) to ask about: {topics}{extra}\n"
        f"Question form: {STAGE_INSTRUCTIONS[decision.stage]}\n"
        f"Style: {GUIDANCE_INSTRUCTIONS[decision.guidance]}"
    )
    attempts: list[dict] = []
    for _ in range(max_drafts):
        messages = [{"role": "system", "content": GENERATION_PROMPT}, {"role": "user", "content": request}]
        try:
            draft = llm.chat(messages).strip().strip('"')
        except LLMError as err:
            attempts.append({"draft": None, "problems": [str(err)]})
            break
        problems = rule_check(draft, patient_text)
        if not problems and use_critic:
            try:
                verdict = llm.structured(
                    [
                        {"role": "system", "content": CRITIC_PROMPT},
                        {"role": "user", "content": f"What the patient has said so far: {patient_text}\nQuestion to check: {draft}"},
                    ],
                    NeutralityVerdict,
                )
                if verdict.leading:
                    problems.append(f"critic: {verdict.reason}")
            except LLMError as err:
                problems.append(f"critic unavailable: {err}")
        if not problems:
            return QuestionResult(draft, "llm", attempts)
        attempts.append({"draft": draft, "problems": problems})
        request += f"\n\nYour previous draft was rejected ({'; '.join(problems)}). Write a neutral question."
    return QuestionResult(template_question(decision, state), "template", attempts)
