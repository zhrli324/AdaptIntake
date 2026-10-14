from adaptintake.clarity import ClarityTracker, Guidance
from adaptintake.extraction import TurnUpdate, apply_analysis, quote_supported
from adaptintake.policy import PolicyConfig, Stage, decide
from adaptintake.questions import rule_check
from adaptintake.redflags import screen
from adaptintake.schema import SLOT_ORDER, InformationState, SlotStatus, SlotUpdate, TurnAnalysis
from adaptintake.summary import build_summary


def upd(slot, value, evidence, kind="answer", vague=False, contradicts=False, no_more=False):
    return SlotUpdate(slot=slot, value=value, evidence=evidence, kind=kind, vague=vague, contradicts_previous=contradicts, no_more=no_more)


def analysis(*updates, off_topic=False, stop=False):
    return TurnAnalysis(updates=list(updates), off_topic=off_topic, wants_to_stop=stop)


# ---- red flags ---------------------------------------------------------------

def test_red_flags_positive():
    assert screen("I've got this crushing pressure in the middle of my chest").category == "chest pain or pressure"
    assert screen("My speech went slurred this morning").category == "stroke-like symptoms"
    assert screen("I suddenly couldn't lift my right arm, it went weak") is not None


def test_red_flags_negated():
    assert screen("No chest pain, nothing like that") is None
    assert screen("I haven't fainted or passed out") is None
    assert screen("I don't get chest pain at all") is None


# ---- extraction --------------------------------------------------------------

def test_quote_supported_rejects_invented_evidence():
    reply = "It started, oh, about two weeks ago I think."
    assert quote_supported("about two weeks ago", reply)
    assert quote_supported("About two weeks ago.", reply)
    assert not quote_supported("since last Christmas", reply)


def test_apply_analysis_transitions():
    state = InformationState()
    reply1 = "It's been going on for a while"
    u = apply_analysis(state, analysis(upd("onset", "a while", "for a while", vague=True)), reply1, 1)
    assert state["onset"].status == SlotStatus.REPORTED and state["onset"].vague
    assert u.vague == ["onset"] and not u.newly_resolved

    reply2 = "Two weeks ago, just after the new tablet"
    u = apply_analysis(state, analysis(upd("onset", "two weeks ago", "Two weeks ago")), reply2, 2)
    assert state["onset"].is_resolved and state["onset"].clarified
    assert u.newly_resolved == ["onset"]

    reply3 = "Actually it was months ago"
    u = apply_analysis(state, analysis(upd("onset", "months ago", "it was months ago", contradicts=True)), reply3, 3)
    assert state["onset"].status == SlotStatus.CONFLICTING
    assert state["onset"].conflicting_values == ["two weeks ago", "months ago"]

    reply4 = "I don't remember the name of that pill"
    apply_analysis(state, analysis(upd("medications", "name unknown", "I don't remember the name", kind="unknown")), reply4, 4)
    assert state["medications"].status == SlotStatus.UNKNOWN and state["medications"].is_resolved

    u = apply_analysis(state, analysis(upd("allergies", "allergic to penicillin", "penicillin gives me a rash")), "No allergies.", 5)
    assert u.rejected and state["allergies"].status == SlotStatus.MISSING


def test_multi_value_slots_accumulate():
    state = InformationState()
    apply_analysis(state, analysis(upd("medications", "amlodipine 5 mg", "amlodipine 5 mg")), "I take amlodipine 5 mg", 1)
    apply_analysis(state, analysis(upd("medications", "paracetamol", "paracetamol")), "and paracetamol", 2)
    assert state["medications"].value == "amlodipine 5 mg; paracetamol"


# ---- clarity -----------------------------------------------------------------

def test_clarity_moves_between_levels():
    tracker = ClarityTracker()
    clear = TurnUpdate(accepted=[None, None], newly_resolved=["onset", "severity"])
    assert tracker.update(clear) == Guidance.CONFIRM
    unclear = TurnUpdate()  # nothing extracted: a non-answer
    tracker.update(unclear)
    assert tracker.update(unclear) == Guidance.SCAFFOLDED
    tracker.update(clear)
    tracker.update(clear)
    assert tracker.level == Guidance.CONFIRM


# ---- policy ------------------------------------------------------------------

def resolve_all(state):
    for name in SLOT_ORDER:
        state[name].status = SlotStatus.REPORTED
        state[name].value = "x"


def test_conflict_is_asked_first_and_uses_clarify_stage():
    state = InformationState()
    state["onset"].status = SlotStatus.CONFLICTING
    state["onset"].conflicting_values = ["2 days", "2 weeks"]
    d = decide(state, Guidance.FOCUSED, 1, PolicyConfig())
    assert d.slots == ["onset"] and d.stage == Stage.CLARIFY


def test_stops_when_nothing_worth_asking():
    state = InformationState()
    resolve_all(state)
    d = decide(state, Guidance.FOCUSED, 3, PolicyConfig())
    assert d.action == "stop"


def test_declined_and_unknown_are_never_reasked():
    state = InformationState()
    resolve_all(state)
    state["medications"].status = SlotStatus.UNKNOWN
    state["past_history"].status = SlotStatus.DECLINED
    assert decide(state, Guidance.SCAFFOLDED, 3, PolicyConfig()).action == "stop"


def test_funnel_and_bundling():
    state = InformationState()
    state["chief_complaint"].status = SlotStatus.REPORTED
    state["chief_complaint"].value = "dizzy"
    confirm = decide(state, Guidance.CONFIRM, 1, PolicyConfig())
    assert confirm.stage == Stage.OPEN and len(confirm.slots) == 2
    focused = decide(state, Guidance.FOCUSED, 1, PolicyConfig())
    assert focused.stage == Stage.FOCUSED and len(focused.slots) == 1
    state[focused.slots[0]].times_asked = 1
    assert decide(state, Guidance.FOCUSED, 2, PolicyConfig()).stage in (Stage.FOCUSED, Stage.CLOSED)


def test_turn_budget():
    assert decide(InformationState(), Guidance.FOCUSED, 20, PolicyConfig(max_turns=20)).action == "stop"


# ---- questions and summary ---------------------------------------------------

def test_rule_check():
    assert rule_check("You've had chest pain, right?", "I feel dizzy")
    assert rule_check("Could this be a stroke?", "my arm feels odd")
    assert not rule_check("Do you take any medicines?", "I feel dizzy")
    assert not rule_check("When did the stroke happen?", "I had a stroke last year")


def test_summary_sections():
    state = InformationState()
    apply_analysis(state, analysis(upd("chief_complaint", "dizzy when standing", "dizzy when I stand")), "I get dizzy when I stand", 1)
    apply_analysis(state, analysis(upd("onset", "a while", "a while", vague=True)), "a while", 2)
    apply_analysis(state, analysis(upd("onset", "two weeks", "two weeks")), "two weeks", 3)
    apply_analysis(state, analysis(upd("past_history", "-", "rather not say", kind="declined")), "I'd rather not say", 4)
    s = build_summary(state)
    assert [i.slot for i in s.told] == ["chief_complaint"]
    assert [i.slot for i in s.clarified] == ["onset"]
    assert "past_history" in [i.slot for i in s.unclear]
    assert "dizzy when I stand" in s.to_markdown()


def test_unknown_detail_keeps_reported_value():
    state = InformationState()
    apply_analysis(state, analysis(upd("medications", "a stomach tablet", "a stomach tablet", vague=True)), "I take a stomach tablet", 1)
    apply_analysis(state, analysis(upd("medications", "cannot remember its name", "can't remember", kind="unknown")), "I can't remember", 2)
    slot = state["medications"]
    assert slot.status == SlotStatus.REPORTED and not slot.vague  # no more asking for that name
    assert slot.value.startswith("a stomach tablet")


def test_list_slots_ask_anything_else_until_closed():
    state = InformationState()
    resolve_all(state)
    apply_analysis(state, analysis(upd("medications", "amlodipine", "amlodipine")), "I take amlodipine", 1)
    assert state["medications"].list_open and not state["medications"].is_resolved
    d = decide(state, Guidance.SCAFFOLDED, 3, PolicyConfig())
    assert d.slots == ["medications"] and d.stage == Stage.FOCUSED
    apply_analysis(state, analysis(upd("medications", "no other medicines", "that's all", no_more=True)), "that's all", 2)
    assert state["medications"].is_resolved and state["medications"].value.endswith("amlodipine")
