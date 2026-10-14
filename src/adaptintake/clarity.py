"""Assess: estimate how clearly the patient is communicating right now, and pick a guidance level.

The estimate uses only what happened in the conversation (never age or other demographics),
and is updated every turn, so a patient can move between levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .extraction import TurnUpdate


class Guidance(str, Enum):
    CONFIRM = "confirm"  # clear communicator: open questions, may cover two topics at once
    FOCUSED = "focused"  # one topic per question, specific wording
    SCAFFOLDED = "scaffolded"  # one short question, closed options with "not sure", time anchors


@dataclass
class ClarityTracker:
    alpha: float = 0.5  # weight of the latest turn in the moving average
    score: float = 0.6  # start neutral-to-positive; the first replies move it quickly
    confirm_at: float = 0.65
    scaffold_below: float = 0.35
    history: list[dict] = field(default_factory=list)

    @staticmethod
    def turn_signal(update: TurnUpdate) -> tuple[float, dict]:
        """Clarity of a single reply in [0, 1], with the signals that produced it."""
        signals = {
            "resolved": len(update.newly_resolved),
            "vague": len(update.vague),
            "conflicts": len(update.new_conflicts),
            "unknown_or_declined": len(update.unknown_or_declined),
            "non_answer": update.non_answer,
            "off_topic": update.off_topic,
        }
        x = 0.4 + 0.3 * min(signals["resolved"], 2)
        x -= 0.25 * min(signals["vague"], 2)
        x -= 0.35 * min(signals["conflicts"], 2)
        x -= 0.1 * min(signals["unknown_or_declined"], 2)
        if update.non_answer:
            x -= 0.35
        if update.off_topic:
            x -= 0.25
        return max(0.0, min(1.0, x)), signals

    def update(self, update: TurnUpdate) -> Guidance:
        x, signals = self.turn_signal(update)
        self.score = self.alpha * x + (1 - self.alpha) * self.score
        level = self.level
        self.history.append({"signal": round(x, 3), "score": round(self.score, 3), "level": level.value, **signals})
        return level

    @property
    def level(self) -> Guidance:
        if self.score >= self.confirm_at:
            return Guidance.CONFIRM
        if self.score < self.scaffold_below:
            return Guidance.SCAFFOLDED
        return Guidance.FOCUSED
