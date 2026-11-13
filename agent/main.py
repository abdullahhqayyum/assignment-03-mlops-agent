"""CLI: python -m agent.main 'request' [--output evidence.json]."""

import argparse
import json
import logging
import sys
from pathlib import Path

from agent.config import Settings
from agent.orchestrator import Agent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evidence-based MLOps operations agent")
    parser.add_argument("request")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--interactive", action="store_true", help="Permit explicit human confirmation prompts"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = Agent(Settings.from_env())
    response = agent.run(args.request)
    result = response.model_dump(mode="json")
    print(response.answer)
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    if args.interactive:
        for evidence in response.evidence:
            if evidence.status == "confirmation_required":
                print("Proposed operation:", json.dumps(evidence.data, indent=2))
                approved = input("Type YES to execute this exact operation: ").strip() == "YES"
                try:
                    confirmation = agent.registry.confirm(evidence.data["confirmation_id"], approved)
                    result.setdefault("confirmed_operations", []).append(confirmation.model_dump(mode="json"))
                    print(confirmation.model_dump_json(indent=2))
                except (ValueError, PermissionError) as exc:
                    print(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
