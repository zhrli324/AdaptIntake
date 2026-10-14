"""The interview loop: Understand -> Assess -> Ask, repeated until a stopping rule fires, then Summarise.

`IntakeAgent(adaptive=True)` is AdaptIntake. `IntakeAgent(adaptive=False)` is baseline B2: the same
pipeline with guidance fixed at "focused". `FixedQuestionnaire` is baseline B1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import redflags
from .clarity import ClarityTracker, Guidance
from .extraction import TurnUpdate, analyse_reply, apply_analysis
from .llm import LLM, LLMError
from .policy import Decision, PolicyConfig, Stage, decide
from .questions import OPENING_QUESTION, TEMPLATES, generate_question
from .schema import SLOT_ORDER, InformationState, TurnAnalysis
from .summary import Summary, build_summary

CLOSING_MESSAGE = (
    "Thank you, that is very helpful. I have put together a short summary of what you told me. "
    "Please read it and tell me if anything should be corrected before it goes to your doctor."
)
STOP_MESSAGE = "Of course, we can stop here. Thank you for your time. I will show you a summary of what we covered."


@dataclass
class AgentTurn:
    text: str
    done: bool
    record: dict = field(default_factory=dict)


class BaseIntake:
    name = "base"

    def __init__(self, llm: LLM):
        self.llm = llm
        self.state = InformationState()
        self.transcript: list[dict] = []
        self.questions: list[dict] = []  # one record per agent question (for metrics and the demo trace)
        self.turn = 0
        self.done = False
        self.end_reason: str | None = None
        self.red_flag: str | None = None
        self.errors: list[str] = []
        self.rejected_extractions: list[dict] = []  # proposals whose quote was not in the reply

    # -- shared steps ---------------------------------------------------------

    def start(self) -> AgentTurn:
        decision = Decision("ask", ["chief_complaint"], Stage.OPEN, Guidance.FOCUSED, reason="opening question")
        return self._ask(OPENING_QUESTION, decision, source="opening")

    def _ask(self, text: str, decision: Decision, source: str, attempts: list | None = None) -> AgentTurn:
        resolved_before = {s: self.state[s].is_resolved for s in decision.slots}
        for s in decision.slots:
            self.state[s].times_asked += 1
        record = {
            "index": len(self.questions),
            "turn": self.turn,
            "text": text,
            "slots": decision.slots,
            "stage": decision.stage.value if decision.stage else None,
            "guidance": decision.guidance.value if decision.guidance else None,
            "scores": decision.scores,
            "reason": decision.reason,
            "source": source,
            "rejected_drafts": attempts or [],
            "already_resolved": resolved_before,
        }
        self.questions.append(record)
        self.transcript.append({"role": "agent", "text": text})
        return AgentTurn(text, False, record)

    def _finish(self, text: str, reason: str) -> AgentTurn:
        self.done = True
        self.end_reason = reason
        self.transcript.append({"role": "agent", "text": text})
        return AgentTurn(text, True, {"end_reason": reason})

    def _understand(self, reply: str) -> TurnUpdate:
        last_question = next((t["text"] for t in reversed(self.transcript[:-1]) if t["role"] == "agent"), "")
        try:
            analysis = analyse_reply(self.llm, self.state, last_question, reply)
        except LLMError as err:
            self.errors.append(f"turn {self.turn}: extraction failed: {err}")
            analysis = TurnAnalysis(updates=[], off_topic=False, wants_to_stop=False)
        update = apply_analysis(self.state, analysis, reply, self.turn)
        self.rejected_extractions += [{"turn": self.turn, **u.model_dump()} for u in update.rejected]
        return update

    def respond(self, reply: str) -> AgentTurn:
        if self.done:
            raise RuntimeError("the interview has already ended")
        self.turn += 1
        self.transcript.append({"role": "patient", "text": reply})
        hit = redflags.screen(reply)
        if hit:
            self.red_flag = hit.category
            self._understand(reply)  # still record what was said, for the summary
            return self._finish(redflags.URGENT_MESSAGE, f"red flag: {hit.category}")
        update = self._understand(reply)
        if update.wants_to_stop:
            return self._finish(STOP_MESSAGE, "patient asked to stop")
        return self._next(update)

    def _next(self, update: TurnUpdate) -> AgentTurn:
        raise NotImplementedError

    def summary(self) -> Summary:
        return build_summary(self.state, self.red_flag)

    def export(self) -> dict:
        """Everything needed to score or replay the conversation."""
        return {
            "system": self.name,
            "transcript": self.transcript,
            "questions": self.questions,
            "state": self.state.model_dump(mode="json"),
            "summary": self.summary().to_dict(),
            "turns": self.turn,
            "end_reason": self.end_reason,
            "red_flag": self.red_flag,
            "rejected_extractions": len(self.rejected_extractions),
            "errors": self.errors,
            "clarity": getattr(getattr(self, "clarity", None), "history", None),
        }


class IntakeAgent(BaseIntake):
    """AdaptIntake (adaptive=True) or baseline B2 (adaptive=False, guidance fixed at 'focused')."""

    def __init__(self, llm: LLM, adaptive: bool = True, policy: PolicyConfig | None = None, use_critic: bool = True):
        super().__init__(llm)
        self.adaptive = adaptive
        self.name = "adaptintake" if adaptive else "b2_nonadaptive"
        self.policy = policy or PolicyConfig()
        self.use_critic = use_critic
        self.clarity = ClarityTracker()

    def _next(self, update: TurnUpdate) -> AgentTurn:
        guidance = self.clarity.update(update) if self.adaptive else Guidance.FOCUSED
        decision = decide(self.state, guidance, len(self.questions), self.policy)
        if decision.action == "stop":
            return self._finish(CLOSING_MESSAGE, decision.reason)
        result = generate_question(self.llm, decision, self.state, self.transcript, use_critic=self.use_critic)
        return self._ask(result.text, decision, result.source, result.attempts)


class FixedQuestionnaire(BaseIntake):
    """Baseline B1: the same topics, in a fixed order and wording, whatever the patient already said.

    It uses the same extraction and summary so that only the questioning strategy differs.
    """

    name = "b1_fixed"

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.queue = [s for s in SLOT_ORDER if s != "chief_complaint"]

    def _next(self, update: TurnUpdate) -> AgentTurn:
        if not self.queue:
            return self._finish(CLOSING_MESSAGE, "questionnaire complete")
        slot = self.queue.pop(0)
        decision = Decision("ask", [slot], Stage.FOCUSED, None, reason="next item in fixed order")
        return self._ask(TEMPLATES[slot][Stage.FOCUSED], decision, source="template")


SYSTEMS = {
    "b1_fixed": lambda llm: FixedQuestionnaire(llm),
    "b2_nonadaptive": lambda llm: IntakeAgent(llm, adaptive=False),
    "adaptintake": lambda llm: IntakeAgent(llm, adaptive=True),
}
