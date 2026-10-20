"""Rule-based adapters from public datasets (kept unchanged in data/raw/) to case cards.

No model is involved: each source sentence or field becomes one fact, and a slot is assigned from the
source field and keyword rules. Only what a patient could know is kept (examination findings,
test results and diagnoses are dropped).

    adaptintake-data --stats              # counts per case set (and the older-adult subset)
    adaptintake-data --export out/        # write the adapted case cards as JSON for inspection

Sources
    data/raw/mts_dialog/   MTS-Dialog (Ben Abacha et al., EACL 2023), CC BY 4.0. "GENHX" (general history)
                           sections of adult patients; surgical/consent notes are excluded.
    data/raw/agentclinic/  AgentClinic-MedQA extended (Schmidgall et al., 2024-2026), MIT. The "Patient_Actor"
                           part of each OSCE case for adult patients.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .cases import Case
from .config import PROJECT_ROOT
from .redflags import screen

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONTEXT = "No social details beyond the case facts are given in the source."

# ---- shared rules ----------------------------------------------------------------------------------

NUMBER_WORDS = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a few|several|a couple of)"
SLOT_RULES: list[tuple[str, re.Pattern]] = [  # first match wins
    ("allergies", re.compile(r"\ballerg", re.I)),
    ("medications", re.compile(
        r"\b(medications?|medicines?|tablets?|pills?|\d+ ?mg\b|inhalers?|insulin|prescribed|started on|treated with|"
        r"over-the-counter|creams?|ointments?|antacids?|ibuprofen|aspirin|acetaminophen|paracetamol|tylenol|benadryl|antibiotics?)", re.I)),
    ("onset", re.compile(
        rf"\b(ago|since|began|begin|beginning|started|starting|onset|first noticed|for the (past|last)|yesterday|this morning|"
        rf"last (night|week|month|year)|{NUMBER_WORDS}[- ](hours?|days?|weeks?|months?|years?)\b)", re.I)),
    ("quality", re.compile(r"\b(sharp|dull|burning|aching|achy|cramping|throbbing|stabbing|pressure-like|colicky|tingling)\b", re.I)),
    ("region", re.compile(r"\b(radiat|spread|shoots?)\b", re.I)),
    ("provocation", re.compile(r"\b(worse|worsen|better|improv|reliev|aggravat|trigger|exacerbat|eases?|after (eating|meals?|exercise|activity))", re.I)),
    ("timing", re.compile(r"\b(intermittent|constant|episod|daily|nightly|times? a (day|week)|every|comes and goes|at night|in the morning|on and off)", re.I)),
    ("severity", re.compile(r"\b(severe|mild|moderate|\d+ ?/ ?10|out of 10|excruciating|unbearable|unable to)\b", re.I)),
    ("functional_change", re.compile(r"\b(daily activities|walking|work|sleep|appetite|weight (loss|gain)|lost .* weight|independen)", re.I)),
    ("past_history", re.compile(r"\b(history of|diagnosed|hypertension|diabetes|surgery|hospitali[sz]|smok|alcohol|drinks?)\b", re.I)),
]
ESSENTIAL = {"chief_complaint", "onset", "severity", "medications", "allergies"}
AGE_SEX = re.compile(r"(\d{1,3})[- ]year[- ]old[\s,]+(?:\w+[\s-]+){0,3}?(male|female|man|woman|lady|gentleman|boy|girl)", re.I)
SEX_WORDS = {"male": "male", "man": "male", "gentleman": "male", "boy": "male", "female": "female", "woman": "female", "lady": "female", "girl": "female"}


ABBREVIATIONS = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|vs|approx|e\.g|i\.e|[A-Z])\.\s")


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    protected = ABBREVIATIONS.sub(lambda m: m.group(0).replace(". ", "\u2024 "), text)  # do not split "Dr. X"
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    return [p.replace("\u2024", ".").strip() for p in parts if len(p.strip()) > 3]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in {"the", "a", "an", "he", "she", "patient", "his", "her", "of", "and"}}


def near_duplicate(text: str, others: list[str], threshold: float = 0.75) -> bool:
    a = _tokens(text)
    return any(a and (len(a & b) / len(a | b)) >= threshold for b in map(_tokens, others))


def guess_slot(sentence: str, default: str) -> str:
    for slot, pattern in SLOT_RULES:
        if pattern.search(sentence):
            return slot
    return default


def parse_age_sex(text: str) -> tuple[int | None, str | None]:
    m = AGE_SEX.search(text or "")
    if m:
        return int(m.group(1)), SEX_WORDS[m.group(2).lower()]
    m = re.search(r"(\d{1,3})[- ]year[- ]old", text or "", re.I)
    age = int(m.group(1)) if m else None
    lowered = f" {str(text).lower()} "
    sex = "female" if re.search(r"\b(she|her|woman|female|lady)\b", lowered) else "male" if re.search(r"\b(he|his|man|male|gentleman)\b", lowered) else None
    return age, sex


def build_case(case_id: str, title: str, source: str, age: int, sex: str, items: list[tuple[str, str]]) -> Case | None:
    """items: (slot, text). Deduplicates, numbers facts, and flags red-flag facts."""
    facts: list[dict] = []
    for slot, text in items:
        if near_duplicate(text, [f["text"] for f in facts]):
            continue
        facts.append({"id": f"F{len(facts) + 1}", "slot": slot, "text": text, "essential": slot in ESSENTIAL, "known": True, "willing": True})
    if len(facts) < 3 or not any(f["slot"] == "chief_complaint" for f in facts):
        return None
    red = next((f for f in facts if screen(f["text"])), None)
    return Case.model_validate({
        "id": case_id,
        "title": title[:90],
        "source": source,
        "profile": {"age": age, "sex": sex or "unspecified", "context": CONTEXT},
        "facts": facts,
        "contradiction": None,
        "red_flag": {"fact_id": red["id"], "description": screen(red["text"]).category} if red else None,
    })


# ---- MTS-Dialog ----------------------------------------------------------------------------------

MTS_FILES = {
    "MTS-Dialog-TrainingSet.csv": "train",
    "MTS-Dialog-ValidationSet.csv": "valid",
    "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv": "test1",
    "MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv": "test2",
}
MTS_EXCLUDE_NOTE = re.compile(r"\b(risks|consent|underwent|procedure|surgery was|operating room|postoperative|kyphoplasty)\b", re.I)
MTS_CLINICIAN_ONLY = re.compile(
    r"\b(we (discussed|recommend|will|did|have|are)|plan\b|was (seen|admitted|given|referred|brought)|on arrival|emergency room|"
    r"EMS\b|CT\b|MRI|x-ray|ultrasound|scan|labs?\b|white count|EKG|ECG|biopsy|exam|vital|heart rate was|blood pressure was|"
    r"I (have|will|discussed)|Dr\. ?\w+ (feels|found|performed|recommend)|upon review|found to (be|have)|serolog|"
    r"positive for|negative for|was noted|were noted|showed|revealed)",
    re.I,
)


def mts_dialog_cases(directory: Path = RAW_DIR / "mts_dialog") -> list[Case]:
    cases = []
    for filename, tag in MTS_FILES.items():
        path = directory / filename
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["section_header"] != "GENHX" or MTS_EXCLUDE_NOTE.search(row["section_text"]):
                    continue
                age, sex = parse_age_sex(row["section_text"])
                if age is None or age < 18:
                    continue
                kept = [s for s in sentences(row["section_text"]) if not MTS_CLINICIAN_ONLY.search(s)]
                if not kept:
                    continue
                items = [("chief_complaint", kept[0])] + [(guess_slot(s, "associated_symptoms"), s) for s in kept[1:]]
                case = build_case(
                    f"mts_{tag}_{row['ID']}",
                    re.sub(r"^(The patient is |This is |Briefly, )", "", kept[0]),
                    f"MTS-Dialog {filename} ID {row['ID']} (GENHX), CC BY 4.0; rule-based adaptation",
                    age, sex, items,
                )
                if case:
                    cases.append(case)
    return cases


# ---- AgentClinic-MedQA -------------------------------------------------------------------------------

AC_FIELDS = {  # Patient_Actor field -> default slot (keyword rules may refine it)
    "History": "chief_complaint",
    "Past_Medical_History": "past_history",
    "Social_History": "past_history",
    "Family_History": "past_history",
    "Review_of_Systems": "associated_symptoms",
    "Current_Medications": "medications",
    "Medications": "medications",
    "Drug_History": "medications",
}
FIXED_SLOT_FIELDS = {"Past_Medical_History", "Social_History", "Family_History", "Review_of_Systems", "Current_Medications", "Medications", "Drug_History"}


def _as_text(value) -> str:
    if isinstance(value, list):
        return " ".join(f"{v.rstrip('.')}." for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k.replace('_', ' ')}: {_as_text(v)}" for k, v in value.items())
    return str(value or "")


def agentclinic_cases(path: Path = RAW_DIR / "agentclinic" / "agentclinic_medqa_extended.jsonl") -> list[Case]:
    if not path.exists():
        return []
    cases = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        actor = json.loads(line)["OSCE_Examination"]["Patient_Actor"]
        age, sex = parse_age_sex(actor.get("Demographics", ""))
        if age is None or age < 18:
            continue
        symptoms = actor.get("Symptoms") or {}
        primary = symptoms.get("Primary_Symptom") if isinstance(symptoms, dict) else None
        items: list[tuple[str, str]] = []
        if primary:
            items.append(("chief_complaint", f"Main problem: {primary}."))
        for s in (symptoms.get("Secondary_Symptoms") or []) if isinstance(symptoms, dict) else []:
            items.append((guess_slot(s, "associated_symptoms"), f"{s.rstrip('.')}."))
        for field, default in AC_FIELDS.items():
            for s in sentences(_as_text(actor.get(field))):
                items.append((default if field in FIXED_SLOT_FIELDS else guess_slot(s, "associated_symptoms"), s))
        case = build_case(
            f"ac_{index:03d}",
            primary or actor.get("Demographics", "AgentClinic case"),
            f"AgentClinic-MedQA extended, line {index} (Patient_Actor only), MIT; rule-based adaptation",
            age, sex, items,
        )
        if case:
            cases.append(case)
    return cases


ADAPTERS = {"mts_dialog": mts_dialog_cases, "agentclinic": agentclinic_cases}


def main(argv: list[str] | None = None) -> None:
    from .cases import CASE_SET_NAMES, load_cases

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true", help="print counts per case set")
    parser.add_argument("--export", type=Path, help="write adapted case cards as JSON to this folder")
    args = parser.parse_args(argv)
    if args.export:
        for name in ADAPTERS:
            out = args.export / name
            out.mkdir(parents=True, exist_ok=True)
            for case in load_cases(name):
                (out / f"case_{case.id}.json").write_text(case.model_dump_json(indent=2))
        print(f"exported to {args.export}")
    if args.stats or not args.export:
        total = older = facts = 0
        print(f"{'case set':14s} {'cases':>6s} {'age 65+':>8s} {'facts':>7s} {'red flags':>10s}")
        for name in CASE_SET_NAMES:
            cs = load_cases(name)
            n_old = sum(c.profile.age >= 65 for c in cs)
            n_facts = sum(len(c.facts) for c in cs)
            print(f"{name:14s} {len(cs):6d} {n_old:8d} {n_facts:7d} {sum(c.red_flag is not None for c in cs):10d}")
            total, older, facts = total + len(cs), older + n_old, facts + n_facts
        print(f"{'total':14s} {total:6d} {older:8d} {facts:7d}")


if __name__ == "__main__":
    main()
