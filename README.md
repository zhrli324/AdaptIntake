# AdaptIntake

An adaptive pre-consultation agent for older patients. It interviews the patient before a visit,
adapts its guidance to how clearly the patient is answering (never to age), and produces a
summary that the patient confirms before it reaches the clinician. It does not diagnose.

## Method

An LLM handles language: it extracts information from each reply and phrases each question.
Explicit state and rules make the decisions: which gap to ask about, how to ask, when to stop, and
what counts as an emergency.

```
patient reply -> red-flag check -> extract (with quotes) -> update state -> assess clarity
             -> rank gaps -> phrase a neutral question -> ... -> summary -> patient confirms
```

| File | Role |
|---|---|
| `schema.py` | Information state: OPQRST symptom slots, history, medicines, allergies, daily function, main concern |
| `extraction.py` | LLM extraction; every update must quote the patient |
| `clarity.py` | Clarity signals that set the guidance level: confirm / focused / scaffolded |
| `policy.py` | Gap priority with an effort penalty, question funnel, stopping rules |
| `questions.py` | Question wording, answer options and neutrality check |
| `recall.py` | Recall cues for medicines, history and other symptoms; conversation-level consolidation |
| `redflags.py` | Keyword screen plus an LLM check that can only veto false alarms |
| `summary.py` | Summary built from the state, with the patient's own words |
| `agent.py` | The loop and the systems compared: `adaptintake`, `non_adaptive`, `fixed_questionnaire` |

Details and pseudocode: [docs/design.md](docs/design.md).

## Data

| Case set | Cases | Age 65+ | Facts | Source |
|---|---|---|---|---|
| `synthetic` | 20 | 20 | 304 | Hand-written synthetic cases |
| `mts_curated` | 5 | 5 | 47 | Hand-transcribed from MTS-Dialog |
| `mts_dialog` | 134 | 40 | 1,098 | [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) (CC BY 4.0), rule-based adaptation |
| `agentclinic` | 165 | 31 | 1,951 | [AgentClinic-MedQA](https://github.com/SamuelSchmidgall/AgentClinic) (MIT), rule-based adaptation |

The public datasets are stored unchanged in `data/raw/` ([sources and licenses](data/raw/README.md)).
A simulated patient plays each case in one of four styles: `clear`, `fragmented`, `uncertain` or
`contradictory`.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add your endpoint and key
pytest                          # offline tests

streamlit run app/streamlit_app.py                     # demo: Talk / Simulate / Replay
adaptintake-eval --run-id trial --cases case_01        # simulate and score
adaptintake-eval --run-id ac65 --case-set agentclinic --min-age 65
adaptintake-analyze --run-id trial                     # tables and figures
adaptintake-data --stats                               # dataset counts
```

The metrics are:

- **Completeness**: the share of case facts captured.
- **Summary fidelity**: the share of summary statements supported by what the patient said.
- **Unnecessary questions**: questions about topics that were already resolved.
- **Number of questions**.
- **Leading-question rate**.
- **Red-flag detection**.

Results are written to `results/`, which is not tracked.
