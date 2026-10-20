# Raw public datasets

These files are kept exactly as published. `src/adaptintake/datasets.py` converts them into case
cards at load time, using rules only (no model): `load_cases("mts_dialog")` and
`load_cases("agentclinic")`.

| Folder | Source | License | What is used |
|---|---|---|---|
| `mts_dialog/` | MTS-Dialog: Ben Abacha, Yim, Fan & Lin, *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*, EACL 2023. https://github.com/abachaa/MTS-Dialog | CC BY 4.0 (`mts_dialog/LICENSE.txt`) | The 4 main CSV files (1,701 dialogue–section pairs); "GENHX" (general history) sections of adult patients, excluding surgical or consent notes |
| `agentclinic/` | AgentClinic: Schmidgall et al., *AgentClinic: a multimodal benchmark for tool-using clinical AI agents*, npj Digital Medicine 9:499 (2026). https://github.com/SamuelSchmidgall/AgentClinic | MIT (`agentclinic/LICENSE.txt`) | `agentclinic_medqa_extended.jsonl` (215 OSCE cases); only the `Patient_Actor` part of adult cases. Examination findings, test results and diagnoses are not used |

The adapted cards are therefore lower-fidelity than the hand-written sets in `data/cases/`:

- the slot of each fact is a keyword guess;
- facts are whole source sentences, sometimes with more than one detail;
- near-duplicates are only partly removed.

Use `adaptintake-data --export out/` to inspect the adapted cards.
