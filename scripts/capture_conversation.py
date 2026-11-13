"""Capture a real LLM conversation and its tool evidence; never substitutes mocked services."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = Agent(Settings.from_env())
    response = agent.run(args.request)
    transcript = {
        "scenario": args.name,
        "mode": "live_services_and_llm",
        "turns": [{"user": args.request, "assistant": response.model_dump()}],
    }
    print(response.answer)
    if args.interactive:
        for item in response.evidence:
            if item.status != "confirmation_required":
                continue
            print(json.dumps(item.data, indent=2))
            approved = input("Type YES to execute this exact operation: ").strip() == "YES"
            confirmation = agent.registry.confirm(item.data["confirmation_id"], approved)
            transcript["turns"].append(
                {
                    "user": "YES: " + json.dumps(item.data["arguments"]),
                    "confirmed_tool_result": confirmation.model_dump(),
                }
            )
            if confirmation.status == "ok" and confirmation.tool == "airflow_trigger":
                dag_id = confirmation.data["dag_id"]
                run_id = confirmation.data["dag_run_id"]
                for _ in range(48):
                    observed = agent.registry.execute(
                        "airflow_run_status", {"dag_id": dag_id, "run_id": run_id}, response.request_id
                    )
                    if observed.status == "error" or observed.data.get("state") in {"success", "failed"}:
                        break
                    time.sleep(5)
                transcript["workflow_observation"] = observed.model_dump()
                followup = f"Report the actual outcome and task statuses of DAG {dag_id}, run {run_id}."
                final = agent.run(followup)
                transcript["turns"].append({"user": followup, "assistant": final.model_dump()})
                print(final.answer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(transcript, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved actual conversation:", args.output)


if __name__ == "__main__":
    main()
