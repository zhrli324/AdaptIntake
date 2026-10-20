"""Rule-based red-flag screen, run on every patient reply before anything else.

Deliberately simple and conservative: a keyword match (without a nearby negation) pauses the
interview and shows an urgent-care message. The list is illustrative for a course prototype and
is NOT a validated clinical triage tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel

RED_FLAGS: dict[str, list[str]] = {
    "chest pain or pressure": [
        r"chest (?:pain|pressure|tightness|heaviness)",
        r"(?:pain|pressure|tightness|heaviness|crushing|squeezing)[\w\s,]{0,30}\bchest\b",
    ],
    "stroke-like symptoms": [
        r"(?:face|mouth) (?:is |was |has )?(?:drooping|droopy|dropped)",
        r"(?:slurred|slurring|garbled) (?:speech|words)|(?:speech|words|talking)[\w\s]{0,15}(?:slurred|slurring|garbled)",
        r"(?:can't|cannot|couldn't|trouble|difficulty) (?:speak|speaking|talk|talking|get (?:my|the) words out)",
        r"(?:sudden(?:ly)?|all of a sudden)[\w\s,']{0,45}(?:weak|weakness|numb|numbness|can't move|cannot move)",
        r"(?:weak|weakness|numb|numbness)[\w\s,]{0,20}(?:one side|left side|right side|left arm|right arm|left leg|right leg)",
    ],
    "severe breathing difficulty": [
        r"(?:can't|cannot|couldn't|struggling to|hard to) (?:breathe|catch my breath)",
        r"(?:gasping|choking) for (?:air|breath)",
    ],
    "loss of consciousness": [r"(?:passed out|fainted|blacked out|lost consciousness|collapsed)"],
    "bleeding": [r"(?:coughing|vomiting|throwing) up blood", r"blood in (?:my )?(?:vomit|stool|poo)", r"black,? (?:tarry )?(?:stools|poo)"],
    "worst-ever headache": [r"worst headache", r"(?:sudden|thunderclap)[\w\s]{0,15}headache"],
    "self-harm": [r"(?:kill|hurt|harm) myself", r"(?:end|take) my (?:own )?life", r"suicid"],
}

NEGATION = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bdenies\b|\bnor\b|n't\b)[\w\s,']{0,25}$")

URGENT_MESSAGE = (
    "Thank you for telling me. What you just described can be a sign of something that needs "
    "urgent attention. I am pausing these questions. Please seek medical help now: call your "
    "local emergency number or tell a member of staff right away."
)


VERIFY_PROMPT = """A keyword screen flagged a possible emergency warning sign ({category}) in a patient's reply
during a pre-consultation interview. Decide whether the reply plausibly describes that warning sign.
Answer emergency=false ONLY if the match is clearly not a warning sign (e.g. a blocked nose, a past event
that has fully resolved, or the phrase used in an unrelated sense). If in any doubt, answer emergency=true."""


class RedFlagVerdict(BaseModel):
    emergency: bool
    reason: str


def verify(llm, hit: "RedFlagHit", reply: str, question: str) -> RedFlagVerdict:
    """Second stage: the rule decides recall, the LLM may only veto clear false positives.

    Any failure keeps the escalation (fail-safe)."""
    try:
        return llm.structured(
            [
                {"role": "system", "content": VERIFY_PROMPT.format(category=hit.category)},
                {"role": "user", "content": f"Question: {question}\nPatient reply: {reply}\nMatched text: {hit.matched_text}"},
            ],
            RedFlagVerdict,
        )
    except Exception as err:  # noqa: BLE001 - any error must escalate
        return RedFlagVerdict(emergency=True, reason=f"verification unavailable ({err}); escalating")


@dataclass
class RedFlagHit:
    category: str
    matched_text: str


def screen(text: str) -> RedFlagHit | None:
    lowered = text.lower().replace("’", "'")
    for category, patterns in RED_FLAGS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                preceding = lowered[max(0, match.start() - 40) : match.start()]
                if NEGATION.search(preceding):
                    continue
                return RedFlagHit(category, match.group(0))
    return None
