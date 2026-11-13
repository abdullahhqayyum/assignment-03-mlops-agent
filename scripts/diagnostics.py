"""Export bounded, redacted service diagnostics; never print environment or kubeconfig contents."""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.logging import Redactor  # noqa: E402


def main():
    values = dotenv_values(ROOT / ".env")
    secret_values = [v for k, v in values.items() if any(s in k for s in ("KEY", "SECRET", "PASSWORD"))]
    redactor = Redactor([*secret_values, os.getenv("OPENROUTER_API_KEY", "")])
    result = {}
    commands = {
        "compose_status": ["docker", "compose", "ps", "--all", "--format", "json"],
        "compose_logs": ["docker", "compose", "logs", "--no-color", "--tail", "100"],
        "airflow_startup": ["docker", "compose", "logs", "--no-color", "--tail", "1000", "airflow"],
        "dag_import_errors": [
            "docker",
            "compose",
            "exec",
            "-T",
            "airflow",
            "airflow",
            "dags",
            "list-import-errors",
            "--output",
            "json",
        ],
        "kubernetes": [
            "kubectl",
            "--kubeconfig",
            ".runtime/kubeconfig",
            "-n",
            "mlops-agent",
            "get",
            "pods,deployments,services,events",
            "-o",
            "json",
        ],
    }
    for name, command in commands.items():
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=25)
            result[name] = {
                "returncode": process.returncode,
                "stdout": process.stdout[:120000],
                "stderr": process.stderr[-3000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"error": type(exc).__name__}
    destination = ROOT / "docs/evidence/service-diagnostics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(redactor.clean(result), indent=2) + "\n")
    print("Redacted service diagnostics saved:", destination)


if __name__ == "__main__":
    main()
