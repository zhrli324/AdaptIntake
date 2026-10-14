"""Summarise: build the clinician summary from the information state only (no free generation).

Three sections: what the patient told us, what was clarified during the conversation, and what is
still unclear. Every statement carries the patient's own words as evidence, so the patient can check
it and the clinician can see where it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import SLOTS, InformationState, SlotStatus


@dataclass
class SummaryItem:
    slot: str
    label: str
    text: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class Summary:
    told: list[SummaryItem] = field(default_factory=list)
    clarified: list[SummaryItem] = field(default_factory=list)
    unclear: list[SummaryItem] = field(default_factory=list)
    red_flag: str | None = None

    def statements(self) -> list[SummaryItem]:
        """Items that assert something the patient said (used for the fidelity metric)."""
        return self.told + self.clarified

    def to_markdown(self) -> str:
        def section(title: str, items: list[SummaryItem], with_quotes: bool) -> list[str]:
            if not items:
                return []
            lines = [f"**{title}**"]
            for item in items:
                line = f"- *{item.label.capitalize()}*: {item.text}"
                if with_quotes and item.evidence:
                    line += "  \n  <small>Your words: " + " / ".join(f"“{q}”" for q in item.evidence) + "</small>"
                lines.append(line)
            return lines + [""]

        lines = []
        if self.red_flag:
            lines += [f"**Urgent:** the interview was paused because of a possible warning sign ({self.red_flag}).", ""]
        lines += section("What you told us", self.told, True)
        lines += section("What we clarified together", self.clarified, True)
        lines += section("Still unclear", self.unclear, False)
        return "\n".join(lines).strip()

    def to_dict(self) -> dict:
        def items(xs):
            return [{"slot": i.slot, "text": i.text, "evidence": i.evidence} for i in xs]

        return {"told": items(self.told), "clarified": items(self.clarified), "unclear": items(self.unclear), "red_flag": self.red_flag}


def build_summary(state: InformationState, red_flag: str | None = None) -> Summary:
    summary = Summary(red_flag=red_flag)
    for name, slot in state.slots.items():
        label = SLOTS[name].label
        quotes = [e.quote for e in slot.evidence]
        if slot.status == SlotStatus.REPORTED:
            text = slot.value or ""
            if slot.vague:
                summary.unclear.append(SummaryItem(name, label, f"Mentioned, but not specific: {text}", quotes))
                continue
            target = summary.clarified if slot.clarified else summary.told
            target.append(SummaryItem(name, label, text, quotes))
        elif slot.status == SlotStatus.CONFLICTING:
            a, b = (slot.conflicting_values + ["", ""])[:2]
            summary.unclear.append(SummaryItem(name, label, f"Two different answers were given: '{a}' and '{b}'", quotes))
        elif slot.status == SlotStatus.UNKNOWN:
            summary.unclear.append(SummaryItem(name, label, "The patient was not sure.", quotes))
        elif slot.status == SlotStatus.DECLINED:
            summary.unclear.append(SummaryItem(name, label, "The patient preferred not to say.", quotes))
        elif SLOTS[name].essential:
            summary.unclear.append(SummaryItem(name, label, "Not discussed."))
    return summary
