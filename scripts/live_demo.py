"""Dedicated real-service validation, usable on Linux/WSL or a disposable CI runner.

Explicit operator flags authorize the bounded training and failure demonstrations.
They are never available as LLM tool arguments. Ordinary CI stays fully offline.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx
from pydantic_core import to_jsonable_python

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from agent.registry import Registry  # noqa: E402
from agent.tools.airflow_tools import AirflowTools  # noqa: E402
from agent.tools.mlflow_tools import MLflowTools  # noqa: E402
from training import train  # noqa: E402

OUT = ROOT / "docs/evidence"


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + ".json")).write_text(
        json.dumps(to_jsonable_python(value), indent=2, allow_nan=False) + "\n"
    )


def command(*args, timeout=300):
    subprocess.run(args, cwd=ROOT, check=True, timeout=timeout)


def wait_for(check, *, timeout=300):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        try:
            result = check()
            if result:
                return result
        except Exception as exc:
            last = type(exc).__name__
        time.sleep(5)
    raise TimeoutError("Service did not become ready: " + str(last))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve-demo-training", action="store_true", required=True)
    parser.add_argument("--approve-demo-failures", action="store_true", required=True)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings.from_env()
    registry = Registry(settings)
    agent = Agent(settings, registry)
    air = AirflowTools(settings)
    ml = MLflowTools(settings)

    def evidence(name, arguments=None):
        result = registry.execute(name, arguments or {}, "dedicated-live-demo")
        if result.status == "error":
            raise RuntimeError(result.model_dump_json())
        return result

    def conversation(name, prompt):
        if not args.with_llm:
            return None
        result = agent.run(prompt)
        save(
            name,
            {
                "scenario": name,
                "mode": "live_services_and_llm",
                "turns": [{"user": prompt, "assistant": result.model_dump()}],
            },
        )
        if result.status == "incomplete":
            raise RuntimeError("Incomplete live conversation: " + result.answer)
        return result

    def trigger(c=1.0, minimum_f1=0.9):
        proposal = evidence("airflow_trigger", {"c": c, "minimum_f1": minimum_f1})
        # Explicit external operator flag authorizes this exact bounded demonstration.
        confirmed = registry.confirm(proposal.data["confirmation_id"], args.approve_demo_training)
        if confirmed.status != "ok":
            raise RuntimeError(confirmed.model_dump_json())
        run_id = confirmed.data["dag_run_id"]
        final = wait_for(
            lambda: (
                r
                if (r := air.run_status("training_pipeline", run_id))["state"] in {"success", "failed"}
                else None
            ),
            timeout=300,
        )
        return run_id, {
            "proposed": proposal.model_dump(),
            "operator_approval": "--approve-demo-training",
            "confirmed": confirmed.model_dump(),
            "final_status": final,
            "tasks": air.tasks("training_pipeline", run_id),
        }

    wait_for(lambda: httpx.get(settings.mlflow_uri + "/health", timeout=5).status_code == 200)
    wait_for(lambda: any(d["dag_id"] == "training_pipeline" for d in air.list_dags()["dags"]))
    command(
        sys.executable,
        "scripts/reproduce_dataset.py",
        "--revision",
        "dataset-v1",
        "--output",
        str(OUT / "dvc-checkout-v1.json"),
    )
    # Four controlled hyperparameter candidates, plus one real Airflow training run.
    for index, c in enumerate((0.1, 1.0, 5.0, 10.0)):
        work = ROOT / ".runtime/training" / f"seed-{index}"
        train.validate(work)
        train.preprocess(work)
        train.train(work, c)
        train.evaluate(work, f"local-seed-C-{c}")
        train.quality_gate(work, 0.7)
    run_id, pipeline = trigger()
    save("airflow-success", pipeline)
    if pipeline["final_status"]["state"] != "success":
        for task in pipeline["tasks"]["failed_tasks"]:
            if task["state"] == "failed":
                save(
                    "airflow-unexpected-failure",
                    air.task_log("training_pipeline", run_id, task["task_id"], task["try_number"]),
                )
        raise RuntimeError("Training DAG did not succeed")

    best = ml.best_run([], limit=5)
    save("mlflow-runs", best)
    command(sys.executable, "scripts/promote_model.py", "--run-id", best["best"]["run_id"])
    command("docker", "build", "-t", "mlops-iris:1.0.0", ".", timeout=600)
    command(
        "kind",
        "create",
        "cluster",
        "--name",
        "assignment03",
        "--kubeconfig",
        ".runtime/kubeconfig",
        "--image",
        "kindest/node:v1.34.0",
        "--wait",
        "180s",
        timeout=900,
    )
    command("kind", "load", "docker-image", "mlops-iris:1.0.0", "--name", "assignment03")
    kube = ["kubectl", "--kubeconfig", ".runtime/kubeconfig"]
    command(*kube, "apply", "-f", "kubernetes/namespace.yaml")
    command(*kube, "apply", "-f", "kubernetes/")
    command(*kube, "-n", settings.namespace, "rollout", "status", "deployment/iris-api", "--timeout=180s")
    sample = {"features": [5.1, 3.5, 1.4, 0.2]}
    predicted = subprocess.run(
        [
            *kube,
            "create",
            "--raw",
            f"/api/v1/namespaces/{settings.namespace}/services/iris-api:80/proxy/predict",
            "-f",
            "-",
        ],
        cwd=ROOT,
        input=json.dumps(sample),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    prediction = json.loads(predicted.stdout)
    assert prediction["class_name"] == "setosa"
    assert prediction["run_id"] == best["best"]["run_id"]
    save(
        "inference-prediction",
        {"input": sample, "response": prediction, "source": "POST /predict through Kubernetes Service proxy"},
    )
    save(
        "kubernetes-healthy",
        {
            name: evidence(name).model_dump()
            for name in (
                "kubernetes_workloads",
                "kubernetes_services",
                "kubernetes_events",
                "kubernetes_app_version",
            )
        },
    )
    conversation("01-dataset-provenance", "Show dataset-v1 provenance and explain how to reproduce it.")
    conversation(
        "02-model-selection",
        "Compare the last five training runs and recommend the best candidate based on F1 score.",
    )
    conversation(
        "03-pipeline-execution",
        f"Report the actual outcome and task statuses of training_pipeline run {run_id}.",
    )
    conversation(
        "04-deployment-diagnosis",
        "Inspect prediction API Kubernetes health, Service, events and Pod logs. Is it available?",
    )
    integrated = conversation(
        "05-multi-tool-reasoning",
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary.",
    )
    if integrated:
        assert {e.source for e in integrated.evidence if e.status == "ok"} >= {
            "dvc",
            "mlflow",
            "airflow",
            "kubernetes",
        }
        assert not any(e.tool in {"airflow_trigger", "kubernetes_restart"} for e in integrated.evidence)

    # Inject a real failed quality gate, inspect logs, then demonstrate successful resolution.
    failed_id, failed = trigger(c=0.001, minimum_f1=1.0)
    assert failed["final_status"]["state"] == "failed"
    failed["logs"] = [
        air.task_log("training_pipeline", failed_id, t["task_id"], t["try_number"])
        for t in failed["tasks"]["failed_tasks"]
        if t["state"] == "failed"
    ]
    save("failure-01-quality-gate", registry.redactor.clean(failed))
    conversation(
        "failure-01-agent",
        f"Investigate failed training_pipeline run {failed_id}. Inspect failed tasks and logs, "
        "explain the root cause, and recommend a resolution without triggering another run.",
    )
    resolved_id, resolved = trigger()
    assert resolved["final_status"]["state"] == "success"
    save("failure-01-resolution", resolved)

    # A bad model path makes new Pods unready. Existing ready replicas retain service during rollout.
    command(
        *kube,
        "-n",
        settings.namespace,
        "set",
        "env",
        "deployment/iris-api",
        "MODEL_PATH=/app/models/missing.joblib",
    )
    wait_for(
        lambda: any(
            c["ready"] is False
            for p in evidence("kubernetes_workloads").data["pods"]
            for c in p["containers"]
        )
    )
    time.sleep(15)
    broken = {
        name: evidence(name).model_dump()
        for name in ("kubernetes_workloads", "kubernetes_events", "kubernetes_services")
    }
    pods = evidence("kubernetes_workloads").data["pods"]
    broken["logs"] = [
        evidence("kubernetes_pod_logs", {"pod": p["name"]}).model_dump()
        for p in pods
        if any(not c["ready"] for c in p["containers"])
    ]
    save("failure-02-readiness", broken)
    conversation(
        "failure-02-agent",
        "Investigate the unhealthy iris-api rollout. Inspect Pod readiness, logs, events and "
        "Service endpoints. Explain the supported root cause and whether existing replicas "
        "still serve traffic. Do not restart anything.",
    )
    command(*kube, "-n", settings.namespace, "set", "env", "deployment/iris-api", "MODEL_PATH-")
    command(*kube, "-n", settings.namespace, "rollout", "status", "deployment/iris-api", "--timeout=180s")
    save("failure-02-resolution", evidence("kubernetes_workloads").model_dump())
    save(
        "live-demo-result",
        {
            "status": "passed",
            "with_llm": args.with_llm,
            "pipeline_run_id": run_id,
            "best_run_id": best["best"]["run_id"],
            "quality_gate_failure_run_id": failed_id,
            "resolution_run_id": resolved_id,
        },
    )


if __name__ == "__main__":
    main()
