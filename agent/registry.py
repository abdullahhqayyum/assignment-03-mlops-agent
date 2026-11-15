"""Structured tool dispatch, error envelopes, bounded confirmation grants and audit logs."""

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from agent import schemas as s
from agent.config import Settings
from agent.logging import Redactor, log_event
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DVCTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MLflowTools


@dataclass
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: object
    mutation: bool = False


class Registry:
    def __init__(self, settings: Settings, tools: list[Tool] | None = None):
        self.settings = settings
        self.redactor = Redactor([settings.openrouter_api_key, settings.airflow_password])
        self.pending = {}
        if tools is None:
            dvc, ml, air, kube = (
                DVCTools(settings),
                MLflowTools(settings),
                AirflowTools(settings),
                KubernetesTools(settings),
            )
            tools = [
                Tool("dvc_list_datasets", "List actual DVC-tracked datasets", s.Empty, dvc.list_datasets),
                Tool(
                    "dvc_dataset_version",
                    "Read committed DVC hash and Git revision; use dataset.git_commit from MLflow",
                    s.DatasetVersion,
                    dvc.dataset_version,
                ),
                Tool(
                    "dvc_compare_versions",
                    "Compare two committed dataset references with dvc diff",
                    s.DatasetDiff,
                    dvc.compare_versions,
                ),
                Tool(
                    "dvc_status",
                    "Detect changed or missing versioned data in the workspace",
                    s.Empty,
                    dvc.status,
                ),
                Tool(
                    "dvc_reproduction",
                    "Explain exact historical checkout commands; does not execute checkout",
                    s.DatasetVersion,
                    dvc.reproduction,
                ),
                Tool(
                    "mlflow_list_experiments",
                    "List MLflow experiment names and identifiers",
                    s.Empty,
                    ml.list_experiments,
                ),
                Tool(
                    "mlflow_list_runs",
                    "List a bounded number of recent runs across specified or all visible experiments",
                    s.Runs,
                    ml.list_runs,
                ),
                Tool(
                    "mlflow_run_details",
                    "Get recorded run metrics, parameters, dataset provenance, tags and artifacts",
                    s.Run,
                    ml.run_details,
                ),
                Tool(
                    "mlflow_best_run",
                    "Rank recent FINISHED runs by metric/direction; check gate before promotion",
                    s.BestRun,
                    ml.best_run,
                ),
                Tool(
                    "mlflow_compare_runs",
                    "Compare two or more runs, including metrics and provenance",
                    s.CompareRuns,
                    ml.compare_runs,
                ),
                Tool(
                    "mlflow_registry",
                    "Retrieve registered model names, aliases and versions",
                    s.Empty,
                    ml.registry,
                ),
                Tool(
                    "airflow_list_dags",
                    "List available DAGs and which DAGs are approved",
                    s.Empty,
                    air.list_dags,
                ),
                Tool(
                    "airflow_list_runs",
                    "Get the most recent Airflow DAG runs and their statuses",
                    s.Dag,
                    air.list_runs,
                ),
                Tool(
                    "airflow_run_status",
                    "Get the recorded state of one DAG run; queued/running is not success",
                    s.DagRun,
                    air.run_status,
                ),
                Tool(
                    "airflow_tasks",
                    "Get task execution details and failed/upstream-failed tasks",
                    s.DagRun,
                    air.tasks,
                ),
                Tool(
                    "airflow_task_log",
                    "Read available task log diagnostics for a specific attempt",
                    s.TaskLog,
                    air.task_log,
                ),
                Tool(
                    "airflow_trigger",
                    "Request approved training only when explicitly asked; requires human confirmation",
                    s.TriggerDag,
                    air.trigger,
                    True,
                ),
                Tool(
                    "kubernetes_workloads",
                    "Inspect inference Pods, Deployment, readiness, container state and restarts",
                    s.Cluster,
                    kube.workload_status,
                ),
                Tool(
                    "kubernetes_services",
                    "Inspect ClusterIP Service selectors, ports and ready/unready endpoints",
                    s.Cluster,
                    kube.services,
                ),
                Tool(
                    "kubernetes_pod_logs", "Read bounded logs for an application Pod", s.PodLog, kube.pod_logs
                ),
                Tool(
                    "kubernetes_events",
                    "Inspect recent application scheduling, probe and container events",
                    s.Cluster,
                    kube.events,
                ),
                Tool(
                    "kubernetes_app_version",
                    "Read running API version, model run ID and dataset provenance through Service proxy",
                    s.Cluster,
                    kube.app_version,
                ),
                Tool(
                    "kubernetes_restart",
                    "Request iris-api restart in approved namespace; requires human confirmation",
                    s.Restart,
                    kube.restart,
                    True,
                ),
            ]
        self.tools = {tool.name: tool for tool in tools}

    def definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.arguments.model_json_schema(),
                },
            }
            for t in self.tools.values()
        ]

    def execute(
        self, name: str, arguments: dict, request_id: str, *, _confirmed: bool = False
    ) -> s.ToolResult:
        start = time.monotonic()
        result = s.ToolResult(
            evidence_id="ev_" + uuid4().hex[:12],
            request_id=request_id,
            tool=name,
            source=name.split("_", 1)[0],
            status="error",
            observed_at=datetime.now(UTC).isoformat(),
            duration_ms=0,
        )
        try:
            if name not in self.tools:
                raise ValueError("Unknown tool")
            tool = self.tools[name]
            values = tool.arguments.model_validate(arguments).model_dump()
            if name.startswith("kubernetes_") and values.get("namespace") is None:
                values["namespace"] = self.settings.namespace
            if name.startswith("kubernetes_") and values.get("namespace") != self.settings.namespace:
                raise PermissionError("Namespace is outside the approved Kubernetes namespace")
            if name == "airflow_trigger" and values["dag_id"] not in self.settings.approved_dags:
                raise PermissionError("DAG is not in APPROVED_DAGS")
            if name == "kubernetes_restart" and values["deployment"] != "iris-api":
                raise PermissionError("Only iris-api may be restarted")
            if tool.mutation and not _confirmed:
                self.pending = {k: v for k, v in self.pending.items() if v[3] > time.monotonic()}
                confirmation_id = uuid4().hex
                self.pending[confirmation_id] = (name, values, request_id, time.monotonic() + 300)
                result.status = "confirmation_required"
                result.data = {
                    "confirmation_id": confirmation_id,
                    "operation": name,
                    "arguments": values,
                    "expires_in_seconds": 300,
                    "executed": False,
                    "instruction": "Human confirmation required via CLI prompt or POST /confirm",
                }
            else:
                result.data = tool.handler(**values)
                result.status = "ok"
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            result.error = {"code": "invalid_request", "message": str(exc)[:1500]}
        except PermissionError as exc:
            result.error = {"code": "unauthorized", "message": str(exc)}
        except (TimeoutError, subprocess.TimeoutExpired, httpx.TimeoutException) as exc:
            result.error = {"code": "timeout", "message": str(exc)[:1000] or "Tool timed out; result unknown"}
        except Exception as exc:
            result.error = {"code": "service_unavailable", "message": str(exc)[:1500]}
        result.duration_ms = round((time.monotonic() - start) * 1000, 2)
        result = s.ToolResult.model_validate(self.redactor.clean(result.model_dump()))
        log_event(
            {
                "request_id": request_id,
                "evidence_id": result.evidence_id,
                "tool": name,
                "status": result.status,
                "duration_ms": result.duration_ms,
                "observed_at": result.observed_at,
                "error": result.error,
            }
        )
        return result

    def confirm(self, confirmation_id: str, approved: bool) -> s.ToolResult:
        grant = self.pending.pop(confirmation_id, None)
        if grant is None or grant[3] < time.monotonic():
            raise ValueError("Unknown, expired or already used confirmation")
        name, arguments, request_id, _ = grant
        if not approved:
            raise PermissionError("Operation declined; no modification executed")
        return self.execute(name, arguments, request_id, _confirmed=True)
