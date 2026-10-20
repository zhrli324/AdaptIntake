"""Aggregate a run into tables and figures.

    adaptintake-analyze --run-id main

Writes results/<run-id>/metrics.csv (one row per conversation), summary.md (tables by system and
persona) and figures/*.png.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..agent import SYSTEM_LABELS
from ..config import RESULTS_DIR
from ..simulator import PERSONAS

SYSTEM_ORDER = list(SYSTEM_LABELS)
# Categorical slots 1-3 of the reference palette (validated as a set; fixed order, never cycled).
COLORS = {"fixed_questionnaire": "#2a78d6", "non_adaptive": "#eb6834", "adaptintake": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

METRICS = {
    "completeness": "Information completeness",
    "essential_completeness": "Essential-fact completeness",
    "fidelity": "Summary fidelity (supported)",
    "accuracy": "Summary accuracy (vs case facts)",
    "questions": "Questions asked",
    "unnecessary_questions": "Unnecessary questions",
    "leading_rate": "Leading-question rate",
    "patient_words": "Patient words",
}
FIGURES = ["completeness", "questions", "unnecessary_questions", "leading_rate"]


def load_run(run_id: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS_DIR / run_id / "conversations").glob("*.json")):
        convo = json.loads(path.read_text())
        m = convo["metrics"]
        rows.append({"system": convo["system"], "case": convo["case"], "persona": convo["persona"], **{k: v for k, v in m.items() if not isinstance(v, (dict, list))}})
    return pd.DataFrame(rows)


def summary_tables(df: pd.DataFrame) -> str:
    ordinary = df[df["red_flag_detected"].isna()] if "red_flag_detected" in df else df
    cols = [c for c in METRICS if c in ordinary]
    overall = ordinary.groupby("system")[cols].mean().reindex(SYSTEM_ORDER).rename(index=SYSTEM_LABELS, columns=METRICS)
    lines = ["# Results", "", f"Conversations: {len(df)} ({len(ordinary)} ordinary cases)", "", "## Overall (mean per conversation)", "", overall.round(3).to_markdown(), ""]
    for metric in [c for c in ("completeness", "questions", "unnecessary_questions", "leading_rate") if c in ordinary]:
        table = ordinary.pivot_table(index="system", columns="persona", values=metric, aggfunc="mean").reindex(index=SYSTEM_ORDER, columns=list(PERSONAS))
        lines += [f"## {METRICS[metric]} by persona", "", table.rename(index=SYSTEM_LABELS).round(3).to_markdown(), ""]
    if "red_flag_detected" in df:
        flags = df[df["red_flag_detected"].notna()]
        if len(flags):
            table = flags.groupby("system")[["red_flag_detected", "red_flag_turn"]].mean().reindex(SYSTEM_ORDER).rename(index=SYSTEM_LABELS)
            lines += ["## Red-flag cases", "", table.round(2).to_markdown(), ""]
    return "\n".join(lines)


def grouped_bar(df: pd.DataFrame, metric: str, path) -> None:
    data = df.groupby(["persona", "system"])[metric].agg(["mean", "sem"]).reset_index()
    personas = [p for p in PERSONAS if p in set(data["persona"])]
    systems = [s for s in SYSTEM_ORDER if s in set(data["system"])]
    width = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for j, system in enumerate(systems):
        sub = data[data["system"] == system].set_index("persona").reindex(personas)
        xs = [i + (j - (len(systems) - 1) / 2) * width for i in range(len(personas))]
        ax.bar(xs, sub["mean"], width=width, color=COLORS[system], edgecolor=SURFACE, linewidth=2, label=SYSTEM_LABELS[system], zorder=2)
        ax.errorbar(xs, sub["mean"], yerr=sub["sem"].fillna(0), fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(range(len(personas)), [p.capitalize() for p in personas], color=INK)
    ax.set_title(f"{METRICS[metric]} by communication style", color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if metric in ("completeness", "essential_completeness", "fidelity", "accuracy", "leading_rate"):
        ax.set_ylim(0, 1.05 if metric != "leading_rate" else None)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncols=len(systems), loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases", nargs="+", help="only these case ids (a matched subset)")
    parser.add_argument("--name", default="", help="suffix for the output files, e.g. 'subset'")
    args = parser.parse_args(argv)
    run_dir = RESULTS_DIR / args.run_id
    df = load_run(args.run_id)
    if args.cases:
        df = df[df["case"].isin(args.cases)]
    if df.empty:
        raise SystemExit(f"no conversations found in {run_dir / 'conversations'}")
    suffix = f"_{args.name}" if args.name else ""
    df.to_csv(run_dir / f"metrics{suffix}.csv", index=False)
    (run_dir / f"summary{suffix}.md").write_text(summary_tables(df))
    fig_dir = run_dir / f"figures{suffix}"
    fig_dir.mkdir(exist_ok=True)
    ordinary = df[df["red_flag_detected"].isna()] if "red_flag_detected" in df else df
    for metric in FIGURES:
        if metric in ordinary and ordinary[metric].notna().any():
            grouped_bar(ordinary, metric, fig_dir / f"{metric}.png")
    print((run_dir / f"summary{suffix}.md").read_text())
    print(f"\nWrote metrics{suffix}.csv, summary{suffix}.md and {fig_dir.name}/ in {run_dir}")


if __name__ == "__main__":
    main()
