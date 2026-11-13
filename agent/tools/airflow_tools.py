"""Authenticated Airflow 3 /api/v2 requests. Only allowlisted DAGs can be triggered."""

from datetime import UTC, datetime
from urllib.parse import quote
from uuid import uuid4

import httpx

from agent.config import Settings


class AirflowTools:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or httpx.Client(base_url=settings.airflow_url, timeout=settings.tool_timeout)
        self._token = None

    def authenticate(self) -> None:
        response = self.client.post(
            "/auth/token",
            json={
                "username": self.settings.airflow_username,
                "password": self.settings.airflow_password,
            },
        )
        response.raise_for_status()
        self._token = response.json()["access_token"]

    def request(self, method: str, path: str, **kwargs) -> dict:
        if not self._token:
            self.authenticate()
        response = self.client.request(
            method, "/api/v2" + path, headers={"Authorization": f"Bearer {self._token}"}, **kwargs
        )
        if response.status_code == 401:
            self.authenticate()
            response = self.client.request(
                method, "/api/v2" + path, headers={"Authorization": f"Bearer {self._token}"}, **kwargs
            )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def dag_path(dag_id: str) -> str:
        return "/dags/" + quote(dag_id, safe="")

    def list_dags(self) -> dict:
        response = self.request("GET", "/dags", params={"limit": 50})
        return {
            "dags": [{"dag_id": d["dag_id"], "is_paused": d["is_paused"]} for d in response["dags"]],
            "total_entries": response["total_entries"],
            "approved_dags": self.settings.approved_dags,
        }

    def list_runs(self, dag_id: str = "training_pipeline") -> dict:
        return self.request(
            "GET", self.dag_path(dag_id) + "/dagRuns", params={"limit": 5, "order_by": "-logical_date"}
        )

    def run_status(self, dag_id: str, run_id: str) -> dict:
        return self.request("GET", self.dag_path(dag_id) + "/dagRuns/" + quote(run_id, safe=""))

    def tasks(self, dag_id: str, run_id: str) -> dict:
        result = self.request(
            "GET",
            self.dag_path(dag_id) + "/dagRuns/" + quote(run_id, safe="") + "/taskInstances",
            params={"limit": 100},
        )
        fields = ("task_id", "state", "try_number", "start_date", "end_date", "duration", "hostname")
        tasks = [{field: task.get(field) for field in fields} for task in result["task_instances"]]
        return {
            "tasks": tasks,
            "failed_tasks": [t for t in tasks if t["state"] in {"failed", "upstream_failed"}],
        }

    def task_log(self, dag_id: str, run_id: str, task_id: str, try_number: int = 1) -> dict:
        path = (
            self.dag_path(dag_id)
            + "/dagRuns/"
            + quote(run_id, safe="")
            + "/taskInstances/"
            + quote(task_id, safe="")
            + f"/logs/{try_number}"
        )
        result = self.request("GET", path, params={"full_content": "true"})
        # Bound log volume without silently claiming that omitted output was inspected.
        content = result.get("content", result)
        text = str(content)
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "task_id": task_id,
            "try_number": try_number,
            "content": text[-12000:],
            "truncated": len(text) > 12000,
        }

    def trigger(
        self,
        dag_id: str = "training_pipeline",
        c: float = 1.0,
        minimum_f1: float = 0.9,
        inject_validation_failure: bool = False,
    ) -> dict:
        if dag_id not in self.settings.approved_dags:
            raise PermissionError("DAG is not in APPROVED_DAGS")
        run_id = "agent_" + uuid4().hex
        return self.request(
            "POST",
            self.dag_path(dag_id) + "/dagRuns",
            json={
                "dag_run_id": run_id,
                "logical_date": datetime.now(UTC).isoformat(),
                "conf": {
                    "c": c,
                    "minimum_f1": minimum_f1,
                    "inject_validation_failure": inject_validation_failure,
                },
            },
        )
