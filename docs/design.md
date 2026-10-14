# Design notes

## Principle

The LLM handles language; explicit state and rules handle decisions. The LLM never controls the
conversation flow and never writes the clinician summary freely. As a result:

- every question can be traced to a named information gap and the rule that selected it
  (`questions[i].reason` and `scores` in each saved conversation);
- a model error can at worst produce a badly worded question, which the neutrality check and
  template fallback catch, or a wrong extraction, which the evidence check and patient
  confirmation catch. It cannot skip the red-flag screen or invent summary content.

## Information state

One record per slot: `status` (missing / reported / conflicting / unknown / declined), `value`,
`vague`, `evidence` (the patient's exact quotes with turn numbers), `times_asked` and `clarified`.

A slot is **resolved** if it is reported and not vague, unknown, or declined. Unknown and declined
slots are never asked again.

Slot parameters (`schema.SLOTS`):

- importance `w_k`
- base burden `b_k`
- essential flag

## Main loop (pseudocode)

```
state <- empty slots; clarity <- 0.6
ask OPENING_QUESTION                              # open question about the main problem
for each patient reply r:
    if red_flag(r): show urgent message; stop      # rules, before anything else
    proposals <- LLM_extract(r, state)             # JSON, Pydantic-validated
    for p in proposals:
        if p.evidence not in r: reject p; continue # no unsupported facts
        merge p into state                         # conflict -> CONFLICTING, vague kept open, ...
    if patient asked to stop: stop
    x <- clarity_signal(turn update)               # resolved, vague, contradictions, non-answer, off-topic
    clarity <- 0.5 * x + 0.5 * clarity             # never uses age
    g <- confirm if clarity >= .65 else scaffolded if clarity < .35 else focused
    for open slot k with times_asked < 3:
        s(k) <- w_k * u(k) - lambda * (b_k * scale(g) + 0.25 * times_asked(k))
        # u: conflicting 1.5, missing 1.0, vague 0.6
    if turns >= T_max or max s(k) < tau: stop
    k* <- argmax s(k)
    stage <- CLARIFY if conflicting else funnel[start(g) + times_asked(k*)]   # open -> focused -> closed
    if g = confirm and stage = open: may add a second missing slot to the same question
    q <- LLM_phrase(k*, stage, g); reject if rule_check(q) or critic(q) says leading; fallback: template
    ask q
summary <- template(state)   # told / clarified / still unclear, with quotes
patient confirms or corrects the summary
```

## Guidance levels

| Level | When | How the agent asks |
|---|---|---|
| confirm | recent replies are clear | open questions; may cover two missing topics in one question; lower weight on effort |
| focused | mixed | one topic per question, specific wording |
| scaffolded | recent replies are vague, conflicting or non-answers | one very short question, closed options with "not sure", everyday time anchors; low-value gaps are dropped sooner |

## Baselines

- **B1 fixed questionnaire**: same opening, then every topic in a fixed order with fixed wording.
  Uses the same extraction and summary, so only the questioning strategy differs.
- **B2 non-adaptive**: identical to AdaptIntake but with guidance fixed at `focused`. This isolates
  the effect of adaptivity.

All three systems share the red-flag screen.

## Parameters to tune (after the pilot)

- `PolicyConfig`: `lam`, `threshold`, `max_asks_per_slot`, `max_turns`.
- `ClarityTracker`: `alpha` and the thresholds.
- `schema.SLOTS`: importance and burden.

Change them in one place and re-run. Results are saved per run id, so runs can be compared.
