"""Recheck an existing real stack and export fresh evidence without replacing its deployment."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from agent.registry import Registry  # noqa: E402


def predict_through_service(settings: Settings, sample: dict) -> dict:
    command = ["kubectl"]
    if settings.kubeconfig:
        command.extend(["--kubeconfig", settings.kubeconfig])
    command.extend(
        [
            "create",
            "--raw",
            f"/api/v1/namespaces/{settings.namespace}/services/iris-api:80/proxy/predict",
            "-f",
            "-",
        ]
    )
    response = subprocess.run(
        command,
        cwd=ROOT,
        input=json.dumps(sample),
        text=True,
        capture_output=True,
        check=True,
        timeout=settings.tool_timeout,
    )
    return json.loads(response.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-training", action="store_true")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/live-verification.json")
    args = parser.parse_args()
    settings = Settings.from_env()
    registry = Registry(settings)
    result = {"observed_at": datetime.now(UTC).isoformat(), "mode": "real_services", "checks": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(registry.redactor.clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def check(name, arguments=None):
        observed = registry.execute(name, arguments or {}, "live-verification")
        result["checks"].setdefault(name, []).append(observed.model_dump(mode="json"))
        save()
        if observed.status == "error":
            raise RuntimeError(observed.model_dump_json())
        print(name, observed.status, flush=True)
        return observed

    def training(arguments, expected):
        proposal = check("airflow_trigger", arguments)
        confirmed = registry.confirm(proposal.data["confirmation_id"], True)
        result.setdefault("training", []).append({"confirmation": confirmed.model_dump(mode="json")})
        save()
        assert confirmed.status == "ok", confirmed.error
        run_id = confirmed.data["dag_run_id"]
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            state = check("airflow_run_status", {"run_id": run_id}).data["state"]
            if state in {"success", "failed"}:
                break
            time.sleep(5)
        else:
            raise TimeoutError("Airflow run did not finish within seven minutes")
        tasks = check("airflow_tasks", {"run_id": run_id}).data
        for task in tasks["failed_tasks"]:
            if task["state"] == "failed":
                check(
                    "airflow_task_log",
                    {"run_id": run_id, "task_id": task["task_id"], "try_number": task["try_number"]},
                )
        assert state == expected, f"Expected {expected}, observed {state}"
        return run_id

    def wait_until(name, ready):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            observed = check(name).data
            if ready(observed):
                return observed
            time.sleep(5)
        raise TimeoutError(f"{name} did not become ready within four minutes")

    try:
        dataset_export = args.output.with_name(args.output.stem + "-dataset.json")
        subprocess.run(
            [
                sys.executable,
                "scripts/reproduce_dataset.py",
                "--revision",
                "dataset-v1",
                "--output",
                str(dataset_export.resolve()),
            ],
            cwd=ROOT,
            check=True,
            timeout=120,
        )
        result["dataset_reproduction"] = json.loads(dataset_export.read_text(encoding="utf-8"))
        for name in ("dvc_list_datasets", "dvc_status", "mlflow_list_experiments"):
            check(name)
        wait_until("airflow_list_dags", lambda d: any(x["dag_id"] == "training_pipeline" for x in d["dags"]))
        best = check("mlflow_best_run").data["best"]
        assert best, "No recent finished model with F1 found"
        check("dvc_dataset_version", {"revision": best["tags"]["dataset.git_commit"]})
        check("mlflow_run_details", {"run_id": best["run_id"]})
        check("mlflow_registry")
        workloads = wait_until(
            "kubernetes_workloads",
            lambda w: w["deployments"] and all(d["ready_replicas"] >= 2 for d in w["deployments"]),
        )
        assert len(check("kubernetes_services").data["ready_endpoints"]) >= 2
        version = check("kubernetes_app_version").data["application"]
        check("kubernetes_events")
        for pod in workloads["pods"]:
            check("kubernetes_pod_logs", {"pod": pod["name"]})
        result["prediction"] = predict_through_service(settings, {"features": [5.1, 3.5, 1.4, 0.2]})
        assert result["prediction"]["class_name"] == "setosa"
        assert result["prediction"]["run_id"] == version["run_id"]
        if args.approve_training:
            result["failure_run_id"] = training({"c": 0.001, "minimum_f1": 1.0}, "failed")
            result["validation_failure_run_id"] = training({"inject_validation_failure": True}, "failed")
            result["success_run_id"] = training({}, "success")
        if args.with_llm:
            response = Agent(settings, registry).run(
                "Find the best recent model, identify the dataset version used to train it, "
                "check the deployment status, and explain whether retraining is necessary."
            )
            result["conversation"] = response.model_dump(mode="json")
            save()
            assert response.status == "complete", response.answer
            assert {e.source for e in response.evidence if e.status == "ok"} >= {
                "dvc",
                "mlflow",
                "airflow",
                "kubernetes",
            }
            assert not any(e.tool in {"airflow_trigger", "kubernetes_restart"} for e in response.evidence)
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    finally:
        save()
        print("Saved real-service verification:", args.output, flush=True)


if __name__ == "__main__":
    main()
