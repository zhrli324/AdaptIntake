"""Chat with the agent in the terminal (useful for a quick check of your .env settings).

    python -m adaptintake.cli [--system adaptintake|b2_nonadaptive|b1_fixed] [--trace]
"""

from __future__ import annotations

import argparse

from .agent import SYSTEMS
from .llm import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="adaptintake", choices=list(SYSTEMS))
    parser.add_argument("--trace", action="store_true", help="print the decision behind each question")
    args = parser.parse_args()

    agent = SYSTEMS[args.system](get_llm("agent"))
    turn = agent.start()
    print(f"\nAgent: {turn.text}")
    while not turn.done:
        try:
            reply = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not reply:
            continue
        turn = agent.respond(reply)
        if args.trace and turn.record.get("slots"):
            r = turn.record
            print(f"  [guidance={r['guidance']} stage={r['stage']} slots={r['slots']} | {r['reason']}]")
        print(f"\nAgent: {turn.text}")
    print("\n" + agent.summary().to_markdown().replace("<small>", "").replace("</small>", ""))


if __name__ == "__main__":
    main()
