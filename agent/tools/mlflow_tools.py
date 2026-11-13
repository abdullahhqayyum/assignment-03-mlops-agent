"""MLflow SDK integration. Ranking is explicit about its bounded recent-run search."""

import math
import os
from datetime import UTC, datetime

from mlflow import MlflowClient

from agent.config import Settings


class MLflowTools:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(int(settings.tool_timeout))
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
        self.client = client or MlflowClient(tracking_uri=settings.mlflow_uri)

    def list_experiments(self) -> dict:
        experiments = self.client.search_experiments(max_results=50)
        return {"experiments": [{"id": e.experiment_id, "name": e.name} for e in experiments], "limit": 50}

    def run_dict(self, run) -> dict:
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }

    def list_runs(self, experiment_ids: list[str], limit: int = 5) -> dict:
        ids = experiment_ids or [e["id"] for e in self.list_experiments()["experiments"]]
        if not ids:
            return {"runs": [], "limit": limit, "experiment_ids": []}
        runs = self.client.search_runs(ids, max_results=limit, order_by=["attributes.start_time DESC"])
        return {"runs": [self.run_dict(run) for run in runs], "limit": limit, "experiment_ids": ids}

    def run_details(self, run_id: str) -> dict:
        run = self.client.get_run(run_id)
        return self.run_dict(run) | {"artifacts": [a.path for a in self.client.list_artifacts(run_id)]}

    def best_run(
        self, experiment_ids: list[str], limit: int = 5, metric: str = "f1", direction: str = "maximize"
    ) -> dict:
        recent = self.list_runs(experiment_ids, limit)
        candidates = [
            r
            for r in recent["runs"]
            if r["status"] == "FINISHED" and math.isfinite(r["metrics"].get(metric, float("nan")))
        ]
        best = sorted(candidates, key=lambda r: r["metrics"][metric], reverse=direction == "maximize")
        selected = best[0] if best else None
        age_days = None
        if selected and selected["start_time"]:
            age_days = (datetime.now(UTC).timestamp() * 1000 - selected["start_time"]) / 86400000
        return {
            **recent,
            "metric": metric,
            "direction": direction,
            "best": selected,
            "candidate_age_days": age_days,
            "freshness_policy_days": self.settings.max_model_age_days,
            "freshness_policy_exceeded": age_days > self.settings.max_model_age_days
            if age_days is not None and self.settings.max_model_age_days is not None
            else None,
            "selection_scope": f"Best finite metric among at most {limit} recent runs, not all history",
            "caution": (
                "Metric ranking alone does not establish production readiness; "
                "check gate, data, workflow and deployment."
            ),
        }

    def compare_runs(self, run_ids: list[str]) -> dict:
        return {"runs": [self.run_details(run_id) for run_id in run_ids]}

    def registry(self) -> dict:
        models = self.client.search_registered_models(max_results=20)
        versions = self.client.search_model_versions(max_results=50)
        return {
            "models": [{"name": m.name, "aliases": m.aliases} for m in models],
            "versions": [
                {"name": v.name, "version": v.version, "run_id": v.run_id, "status": v.status}
                for v in versions
            ],
            "limits": {"models": 20, "versions": 50},
        }
