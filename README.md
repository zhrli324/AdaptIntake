# AdaptIntake

An adaptive pre-consultation agent for older patients: a course prototype.

Before a consultation, AdaptIntake interviews the patient about their symptoms, history and
medicines. It adapts how much guidance it gives to how clearly the patient is communicating
**right now**, not to their age. It then produces a summary that the patient checks before it
reaches the clinician. It does not diagnose and does not give medical advice.

> Research prototype for a course project. Not a medical device. Use only synthetic data;
> never enter real patient information.

## How it works

The system is hybrid. The LLM handles language: it extracts information from each reply and
phrases each question. Explicit state and rules make every decision: what to ask, how to ask it,
when to stop, and what counts as an emergency. As a result, every question can be traced to a
named information gap and the rule that selected it.

```
patient reply
  -> red-flag screen (rules)            -> urgent-care message, stop
  -> Understand (LLM extraction)        -> slot updates, each with the patient's exact words
  -> state update (rules)               -> reported / missing / conflicting / unknown / declined
  -> Assess (clarity signals)           -> guidance: confirm / focused / scaffolded
  -> Ask: policy (rules)                -> highest-priority gap, or stop
  -> Ask: wording (LLM + neutrality check, template fallback)
  -> next question ... until stop -> Summarise (template, from state only) -> patient confirms
```

| Module | File | Method |
|---|---|---|
| Information state | `schema.py` | OPQRST slots + history, medicines, allergies, daily function, main concern; status and evidence per slot |
| Understand | `extraction.py` | Structured LLM extraction (Pydantic-validated JSON). Updates whose quote is not in the reply are rejected |
| Assess | `clarity.py` | Per-turn clarity signals (resolved / vague / contradictions / non-answers / off-topic), moving average -> guidance level |
| Ask (decide) | `policy.py` | Priority `s(k) = w_k * u(k) - lambda * b_k(g)`; funnel open -> focused -> closed; conflicts clarified first; stopping rules |
| Ask (phrase) | `questions.py` | Constrained LLM generation, rule check plus LLM critic, neutral templates as fallback |
| Safety | `redflags.py` | Rule-based red-flag screen on every reply (illustrative list, not validated triage) |
| Summarise | `summary.py` | Template summary: what you told us / what we clarified / still unclear, with quotes |
| Loop and baselines | `agent.py` | `IntakeAgent(adaptive=True)` = AdaptIntake; `adaptive=False` = B2; `FixedQuestionnaire` = B1 |

**AdaptIntake v2** (`adaptintake_v2`) adds three mechanisms, each motivated by an error analysis of
the first evaluation run (`recall.py`, `redflags.verify`):

- **Anchored recall**: list follow-ups (medicines, history, other symptoms) use concrete recall cues,
  as in clinical medication reconciliation. Clear communicators get all cues in one question; others
  get one cue at a time.
- **Conversation-level consolidation**: before stopping, the whole conversation is re-read once to
  recover information said in passing. Each addition must quote the patient.
- **Two-stage red-flag check**: the rules give high recall. An LLM may only veto clear false positives,
  and any failure escalates.

More detail and pseudocode: [docs/design.md](docs/design.md).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` with your endpoint, key and model names. Any OpenAI-compatible endpoint works,
including relay services. Use a judge from a different model family than the agent. `.env` is
git-ignored, so never commit keys.

Run the tests (offline, no API calls):

```bash
pytest
```

## Try it

```bash
streamlit run app/streamlit_app.py      # large-text chat demo with an "agent view" panel
python -m adaptintake.cli --trace       # terminal chat; prints the decision behind each question
```

## Evaluation

Twenty synthetic case cards in `data/cases/`: 18 ordinary presentations and 2 red-flag cases. Each
card lists the facts the simulated patient knows. Some facts are marked as not known to the
patient, or as something they prefer not to discuss, and most cards include one planted
contradiction.

An LLM plays each case in four communication styles: `clear`, `fragmented`, `uncertain` and
`contradictory`. Three systems are compared on the same cases:

- **B1**: fixed questionnaire (same topics, fixed order and wording).
- **B2**: the same LLM pipeline with guidance fixed at `focused` (ablation of adaptivity).
- **AdaptIntake**.

```bash
# small pilot first (6 conversations)
adaptintake-eval --run-id pilot --cases case_01 --personas clear uncertain --systems b2_nonadaptive adaptintake
# full run: 20 cases x 4 personas x 3 systems = 240 conversations; resumable
adaptintake-eval --run-id main --workers 6
adaptintake-analyze --run-id main
```

Outputs go to `results/<run-id>/`:

- `conversations/*.json`: transcript, every decision, final state, summary and metrics.
- `metrics.csv`.
- `summary.md`: tables by system and persona.
- `figures/*.png`.

| Metric | How it is measured |
|---|---|
| Completeness | Share of case facts (known and willing) captured in the final state, LLM judge |
| Essential completeness | Same, essential facts only |
| Summary fidelity | Share of summary statements supported by what the patient said, LLM judge |
| Summary accuracy | Share of summary statements consistent with the true case facts, LLM judge |
| Questions | Number of agent questions |
| Unnecessary questions | Questions about topics already resolved (answered, unknown or declined), deterministic |
| Leading-question rate | Questions judged leading (the opening question is excluded), LLM judge; spot-check 20% by hand |
| Red-flag handling | Detected or not, and on which turn (cases 19-20) |

### External validation cases (public data)

`data/cases_external/` holds 5 case cards for patients aged 66-76. They were adapted from
[MTS-Dialog](https://github.com/abachaa/MTS-Dialog) (Ben Abacha et al., EACL 2023, CC BY 4.0) by
transcribing the facts in each source note and dialogue, adding nothing. Each card names its source
record ID. To re-create them, download the CSV files from the MTS-Dialog repository into
`data/external/mts_dialog/`; that folder is git-ignored.

```bash
adaptintake-eval --run-id external --cases-dir data/cases_external --personas clear fragmented
```

`scripts/smoke_eval.py` runs the whole pipeline with fake models, to check that it works without
spending API credit.

**Cost note:** each conversation makes about 3 agent calls and 1 patient call per turn, plus 3
judge calls. Run a pilot first and check the transcripts before the full run.

## Repository layout

```
src/adaptintake/      agent package (modules above), simulator.py, cases.py, llm.py, config.py, cli.py
src/adaptintake/evaluation/   run.py (simulate + score), metrics.py, judge.py, analyze.py
app/streamlit_app.py  demo interface
data/cases/           synthetic case cards (JSON)
tests/                offline tests with a scripted model
docs/design.md        design notes and pseudocode
results/              evaluation outputs (git-ignored)
```

## Limitations

- Simulated patients are not older adults. The evaluation tests whether the design is feasible,
  not whether older adults find it usable.
- The LLM judge can be wrong. Spot-check a sample by hand.
- The red-flag list is illustrative and keyword-based.
- A real deployment would need a locally hosted model so that patient data stays on site.
