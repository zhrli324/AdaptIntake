"""Run systems x cases x personas with simulated patients, then score each conversation.

Examples:
    adaptintake-eval --run-id trial --cases case_01 case_02 --personas clear fragmented
    adaptintake-eval --run-id synthetic --workers 8
    adaptintake-eval --run-id mts --case-set mts_curated --personas clear fragmented
    adaptintake-eval --run-id ac65 --case-set agentclinic --min-age 65 --limit 20
Finished conversations are saved one file each and skipped on re-run, so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from ..agent import SYSTEMS
from ..cases import CASE_SET_NAMES, Case, load_cases
from ..config import RESULTS_DIR, model_settings
from ..llm import LLM, get_llm
from ..simulator import PERSONAS, SimulatedPatient
from .metrics import deterministic_metrics, judge_metrics

MAX_EXCHANGES = 30  # hard safety limit on patient replies per conversation


def simulate(system: str, case: Case, persona: str, agent_llm: LLM, patient_llm: LLM) -> dict:
    agent = SYSTEMS[system](agent_llm)
    patient = SimulatedPatient(patient_llm, case, persona)
    turn = agent.start()
    for _ in range(MAX_EXCHANGES):
        reply = patient.reply(turn.text)
        turn = agent.respond(reply)
        if turn.done:
            break
    else:
        agent.end_reason = "exchange limit"
    convo = agent.export()
    convo.update({"case": case.id, "persona": persona})
    return convo


def run_one(system: str, case: Case, persona: str, out_dir: Path, llms: dict, skip_judge: bool) -> str:
    path = out_dir / f"{system}__{case.id}__{persona}.json"
    if path.exists():
        return f"skip {path.name}"
    convo = simulate(system, case, persona, llms["agent"], llms["patient"])
    convo["metrics"] = deterministic_metrics(case, convo)
    if not skip_judge:
        convo["metrics"].update(judge_metrics(llms["judge"], case, convo))
    path.write_text(json.dumps(convo, ensure_ascii=False, indent=1))
    return f"done {path.name} ({convo['metrics']['questions']} questions, end: {convo['end_reason']})"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--personas", nargs="+", default=list(PERSONAS), choices=list(PERSONAS))
    parser.add_argument("--cases", nargs="+", help="case ids, e.g. case_01 case_19 (default: all)")
    parser.add_argument("--case-set", default="synthetic", choices=CASE_SET_NAMES, help="which case cards to use")
    parser.add_argument("--min-age", type=int, help="only cases at least this old, e.g. 65")
    parser.add_argument("--limit", type=int, help="only the first N cases (after filtering)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-judge", action="store_true", help="only simulate; no judge calls")
    args = parser.parse_args(argv)

    cases = load_cases(args.case_set, ids=args.cases, min_age=args.min_age)[: args.limit]
    out_dir = RESULTS_DIR / args.run_id / "conversations"
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = ("agent", "patient") if args.skip_judge else ("agent", "patient", "judge")
    llms = {role: get_llm(role) for role in roles}
    config = {role: model_settings(role).model for role in roles}
    (out_dir.parent / "config.json").write_text(json.dumps({"models": config, **vars(args)}, indent=1, default=str))

    jobs = [(s, c, p) for c in cases for p in args.personas for s in args.systems]
    print(f"{len(jobs)} conversations -> {out_dir} (models: {config})")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, s, c, p, out_dir, llms, args.skip_judge): (s, c.id, p) for s, c, p in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                print(f"[{i}/{len(jobs)}] {future.result()}", flush=True)
            except Exception:
                print(f"[{i}/{len(jobs)}] FAILED {futures[future]}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
