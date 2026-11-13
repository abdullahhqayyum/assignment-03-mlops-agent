"""Validated local settings; no credentials are included in public tool schemas."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    repo: Path = ROOT
    openrouter_api_key: str = field(default="", repr=False)
    model: str = "openrouter/free"
    mlflow_uri: str = "http://localhost:5000"
    airflow_url: str = "http://localhost:8080"
    airflow_username: str = "admin"
    airflow_password: str = field(default="", repr=False)
    approved_dags: tuple[str, ...] = ("training_pipeline",)
    namespace: str = "mlops-agent"
    kubeconfig: str | None = None
    tool_timeout: float = 15
    llm_timeout: float = 90
    max_steps: int = 16
    max_model_age_days: int | None = None

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env")
        age = os.getenv("MAX_MODEL_AGE_DAYS", "")
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "openrouter/free"),
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            airflow_url=os.getenv("AIRFLOW_URL", "http://localhost:8080").rstrip("/"),
            airflow_username=os.getenv("AIRFLOW_USERNAME", "admin"),
            airflow_password=os.getenv("AIRFLOW_PASSWORD", ""),
            approved_dags=tuple(filter(None, os.getenv("APPROVED_DAGS", "training_pipeline").split(","))),
            namespace=os.getenv("KUBERNETES_NAMESPACE", "mlops-agent"),
            kubeconfig=os.getenv("KUBECONFIG") or None,
            tool_timeout=float(os.getenv("TOOL_TIMEOUT_SECONDS", "15")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
            max_steps=int(os.getenv("MAX_AGENT_STEPS", "16")),
            max_model_age_days=int(age) if age else None,
        )
