# AdaptIntake

An adaptive pre-consultation agent for older patients.

Before a consultation, AdaptIntake interviews the patient about their symptoms, history and
medicines. It adapts how much guidance it gives to how clearly the patient is communicating
**right now**, never to their age. It then produces a summary that the patient checks before it
reaches the clinician. It does not diagnose and does not give medical advice.

> Course prototype. Not a medical device. Use only synthetic data; never enter real patient
> information.

## Method

The system is hybrid. An LLM handles language: it extracts information from each reply and phrases
each question. Explicit state and rules make every decision: what to ask, how to ask it, when to
stop, and what counts as an emergency. As a result, every question can be traced to a named
information gap and the rule that selected it.

```
patient reply
  -> red-flag check     rules (high recall) + LLM veto of clear false alarms  -> urgent message, stop
  -> Understand         LLM extraction; each update must quote the patient
  -> state update       reported / missing / conflicting / unknown / declined
  -> Assess             clarity signals -> guidance: confirm / focused / scaffolded
  -> Ask: decide        gap priority s = w * u - lambda * effort; question funnel; recall cues for lists
  -> Ask: phrase        LLM wording + neutrality check, neutral template as fallback
  ... until no gap is worth the patient's effort -> re-read the conversation once -> stop
  -> Summarise          template, from the state only -> patient confirms or corrects
```

| Module | File | What it does |
|---|---|---|
| Information state | `schema.py` | OPQRST symptom slots plus history, medicines, allergies, daily function and main concern; status and evidence for each slot |
| Understand | `extraction.py` | Structured LLM extraction (Pydantic-validated JSON); updates whose quote is not in the reply are rejected |
| Assess | `clarity.py` | Per-turn clarity signals feed a moving average, which sets the guidance level |
| Ask: decide | `policy.py` | Gap priority with an effort penalty; funnel open -> focused -> closed; conflicts first; stopping rules |
| Ask: phrase | `questions.py` | Constrained generation; closed questions come with answer options and "not sure"; rule check and LLM critic |
| Recall support | `recall.py` | Anchored recall cues for medicines, history and other symptoms; conversation-level consolidation |
| Safety | `redflags.py` | Keyword screen with negation handling, then an LLM check that can only veto; any error escalates |
| Summarise | `summary.py` | Template summary in three parts (what you told us / what we clarified / still unclear), with the patient's words |
| Loop and systems | `agent.py` | `adaptintake`, plus two baselines: `non_adaptive` and `fixed_questionnaire` |

Design notes and pseudocode: [docs/design.md](docs/design.md).

## Data

- `data/cases/synthetic/`: 20 synthetic case cards for patients aged 66–90, with multimorbidity and
  polypharmacy.
  - 18 ordinary presentations and 2 red-flag cases.
  - Some facts are marked as unknown to the patient, or as something the patient prefers not to
    discuss.
  - Most cards plant one contradiction.
- `data/cases/mts_dialog/`: 5 case cards for patients aged 66–76, adapted from
  [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) (Ben Abacha et al., EACL 2023, CC BY 4.0).
  - Each card was transcribed from the source note and dialogue, with nothing added.
  - Each card names its source record ID.

Each case card lists the facts a simulated patient knows. These facts are the ground truth for
evaluation. An LLM plays the patient in one of four communication styles (`simulator.py`): `clear`,
`fragmented`, `uncertain` or `contradictory`.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` with your endpoint, key and model names. Any OpenAI-compatible endpoint works. `.env` is
git-ignored.

Run the tests (offline; they use scripted models):

```bash
pytest
```

## Demo

```bash
streamlit run app/streamlit_app.py
```

The demo has three modes, selected in the sidebar:

- **Talk**: you are the patient. Type an answer, or tap an answer button for a closed question.
  The right-hand panel shows how the agent decides: guidance level, question form, gap priorities
  and the state of every topic.
- **Simulate**: pick a case, a communication style and a system, then watch the interview turn by
  turn. Finished conversations are saved to `results/demos/`.
- **Replay**: step through any saved conversation with a slider. It makes no API calls. You can
  show two conversations side by side, for example the fixed questionnaire and AdaptIntake on the
  same case.

For a quick terminal check: `python -m adaptintake.cli --trace`.

## Evaluation

```bash
adaptintake-eval --run-id trial --cases case_01 --personas clear fragmented        # small trial
adaptintake-eval --run-id synthetic --workers 6                                     # 20 cases x 4 styles x 3 systems
adaptintake-eval --run-id mts --case-set mts_dialog --personas clear fragmented
adaptintake-analyze --run-id synthetic
```

Each conversation is saved as JSON in `results/<run-id>/conversations/`. A saved conversation holds
the transcript, every decision, the final state, the summary and the metrics. Runs are resumable.
`adaptintake-analyze` writes `metrics.csv`, `summary.md` (tables by system and style) and
`figures/`.

| Metric | How it is measured |
|---|---|
| Completeness | Share of case facts (known and willing) captured in the final state, scored by an LLM judge |
| Essential completeness | The same, for essential facts only |
| Summary fidelity | Share of summary statements supported by what the patient said, scored by an LLM judge |
| Summary accuracy | Share of summary statements consistent with the true case facts, scored by an LLM judge |
| Questions | Number of agent questions |
| Unnecessary questions | Questions about topics already resolved; deterministic |
| Leading-question rate | Questions judged leading, excluding the opening question; scored by an LLM judge |
| Red-flag handling | Whether the red flag was detected, and on which turn (red-flag cases) |

Each conversation makes about 3–4 agent calls and 1 patient call per turn, plus 3 judge calls.
Run a small trial first.

## Repository layout

```
src/adaptintake/              method: schema, extraction, clarity, policy, questions, recall,
                              redflags, summary, agent; plus simulator, cases, llm, config, cli
src/adaptintake/evaluation/   run.py (simulate + score), metrics.py, judge.py, analyze.py
app/streamlit_app.py          demo (Talk / Simulate / Replay)
data/cases/                   case cards: synthetic/ and mts_dialog/
docs/design.md                design notes and pseudocode
tests/                        offline tests with scripted models
```

## Limitations

- Simulated patients are not older adults. The evaluation tests whether the design is feasible,
  not whether older adults find it usable.
- LLM judges can be wrong. Spot-check a sample by hand.
- The red-flag keywords are illustrative.
- A real deployment would need a locally hosted model, so that patient data stays on site.
