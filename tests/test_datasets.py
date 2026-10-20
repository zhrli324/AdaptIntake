"""Rule-based adapters for the public datasets in data/raw/ (no network, no model)."""

from adaptintake.cases import CASE_SET_NAMES, load_cases
from adaptintake.datasets import guess_slot, parse_age_sex, sentences


def test_all_case_sets_load_and_are_valid():
    sizes = {name: len(load_cases(name)) for name in CASE_SET_NAMES}
    assert sizes["synthetic"] == 20 and sizes["mts_curated"] == 5
    assert sizes["mts_dialog"] >= 100 and sizes["agentclinic"] >= 150
    assert sum(sizes.values()) >= 300
    for name in ("mts_dialog", "agentclinic"):
        for case in load_cases(name):
            assert case.profile.age >= 18
            assert [f.id for f in case.facts] == [f"F{i}" for i in range(1, len(case.facts) + 1)]
            assert any(f.slot == "chief_complaint" for f in case.facts)
            assert "rule-based adaptation" in case.source


def test_older_subset_filter():
    older = load_cases("agentclinic", min_age=65)
    assert older and all(c.profile.age >= 65 for c in older)


def test_no_examination_or_test_results_leak_into_mts_cases():
    texts = " ".join(f.text for c in load_cases("mts_dialog") for f in c.facts)
    for phrase in ("CT scan showed", "white count", "on arrival", "We discussed"):
        assert phrase not in texts


def test_rules():
    assert sentences("Saw Dr. Smith last week. It hurts.") == ["Saw Dr. Smith last week.", "It hurts."]
    assert parse_age_sex("The patient is a 76-year-old female who") == (76, "female")
    assert parse_age_sex("69-year-old Caucasian male complaining") == (69, "male")
    assert guess_slot("The pain started two weeks ago.", "x") == "onset"
    assert guess_slot("She takes metformin 500 mg twice daily.", "x") == "medications"
    assert guess_slot("He is allergic to penicillin.", "x") == "allergies"
    assert guess_slot("It is a dull, aching pain.", "x") == "quality"
