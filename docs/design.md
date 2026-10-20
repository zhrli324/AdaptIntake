# Design

## Principle

The LLM handles language; explicit state and rules make the decisions. The LLM never controls the
conversation flow and never writes the clinician summary freely. As a result:

- every question can be traced to a named information gap and the rule that selected it
  (`questions[i].reason` and `scores` in each saved conversation);
- a model error can at worst produce a badly worded question or a wrong extraction:
  - a badly worded question is caught by the neutrality check and the template fallback;
  - a wrong extraction is caught by the evidence check and by patient confirmation.

  It cannot skip the red-flag screen or put content into the summary that the patient did not say.

## Information state

There is one record per slot. Each record holds:

- `status`: missing / reported / conflicting / unknown / declined
- `value`
- `vague`
- `evidence`: the patient's exact quotes, with turn numbers
- `times_asked`
- `clarified`
- for list slots (medicines, past history, other symptoms): `list_open` and `anchors_used`

A slot is **resolved** if it is reported (not vague, and with no list still open), unknown, or
declined. Unknown and declined slots are never asked again.

Each slot has these parameters (`schema.SLOTS`): importance `w_k`, base effort `b_k`, and an
essential flag.

## Main loop (pseudocode)

```
state <- empty slots; clarity <- 0.6
ask OPENING_QUESTION                                  # open question about the main problem
for each patient reply r:
    if red_flag_rules(r) and LLM_confirms(r):          # rules: recall; LLM may only veto; errors escalate
        show urgent message; stop
    proposals <- LLM_extract(r, state)                 # JSON, Pydantic-validated
    for p in proposals:
        if p.evidence not in r: reject p; continue     # no unsupported facts
        merge p into state                             # conflict -> CONFLICTING, vague kept open, lists accumulate
    if patient asked to stop: stop
    x <- clarity_signal(turn update)                   # resolved, vague, contradictions, non-answer, off-topic
    clarity <- 0.5 * x + 0.5 * clarity                 # never uses age
    g <- confirm if clarity >= .65 else scaffolded if clarity < .35 else focused
    for each open slot k below its ask limit:
        s(k) <- w_k * u(k) - lambda * (b_k * scale(g) + 0.2 * times_asked(k))
        # u: conflicting 1.5, missing 1.0, list may be incomplete 0.8, vague 0.6
        # cue-based list follow-ups count as half the effort
    if turns >= T_max or max s(k) < tau:
        consolidate(whole conversation)                 # recover facts said in passing (quote-verified)
        if nothing new: stop
    k* <- argmax s(k)
    stage <- CLARIFY if conflicting
             FOCUSED if list follow-up
             funnel[start(g) + times_asked(k*)]         # open -> focused -> closed (with "not sure")
    if list follow-up: pick recall cue(s)               # all cues at once if g = confirm, else one
    if g = confirm and stage = open: may add a second missing slot to the same question
    q <- LLM_phrase(k*, stage, g, cues); reject if rule_check(q) or critic(q) says leading; fallback: template
    ask q
summary <- template(state)   # what you told us / what we clarified / still unclear, with quotes
patient confirms or corrects the summary
```

## Guidance levels

| Level | When | How the agent asks |
|---|---|---|
| confirm | recent replies are clear | open questions; may cover two missing topics in one question; all recall cues in one question; lower weight on effort |
| focused | mixed | one topic per question, specific wording, one recall cue at a time |
| scaffolded | recent replies are vague, conflicting or non-answers | one very short question, closed answer options with "not sure", everyday time anchors; low-value gaps are dropped sooner |

## Recall support (`recall.py`)

- **Anchored recall.** List follow-ups use concrete cues, modelled on clinical medication
  reconciliation. For medicines, the cues are:
  - medicines for each condition the patient mentioned;
  - by time of day;
  - by form (inhalers, drops, creams, injections);
  - medicines bought without a prescription.

  Past history and other symptoms have similar cues. Each cue either refers to something the
  patient already said or names a neutral category, so it does not suggest a new symptom or
  diagnosis.
- **Conversation-level consolidation.** Before stopping, the agent re-reads the whole conversation
  once, to recover facts that were mentioned in passing (for example, a medicine mentioned while
  answering a different question). Every addition must quote one patient reply, and it can never
  create a conflict.

## Safety (`redflags.py`)

The red-flag check has two stages:

1. A keyword screen with negation handling runs on every reply. Its job is high recall.
2. On a hit, an LLM decides whether the reply really describes the warning sign, for example "can't
   breathe through my nose" or "no chest pain" do not. The LLM may only veto a clear false positive.
   Any uncertainty or error escalates.

## Baselines

- **Fixed questionnaire**: the same opening, then every topic in a fixed order with fixed wording.
  It uses the same extraction and summary, so only the questioning strategy differs.
- **Non-adaptive LLM**: the same extraction, gap ranking, question wording and neutrality check, but
  guidance is fixed at `focused`. It has no recall support and no red-flag verification.

All three systems share the rule-based red-flag screen.

## Parameters

- `PolicyConfig`: `lam`, `threshold`, `max_asks_per_slot`, `max_asks_list_slot`, `max_turns`,
  `anchored_burden`.
- `ClarityTracker`: `alpha` and the level thresholds.
- `schema.SLOTS`: importance and effort per slot.
- `IntakeAgent(...)` switches: `adaptive`, `anchored_recall`, `consolidate`, `verify_red_flags`. Use
  these for ablations.
