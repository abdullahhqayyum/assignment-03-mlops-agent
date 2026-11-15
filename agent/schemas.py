"""Tool arguments reject unknown fields and constrain identifiers before adapter execution."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(min_length=1, max_length=250, pattern=r"^[a-zA-Z0-9_.:+@=-]+$")]
Revision = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./~-]*$")]
Namespace = Annotated[str, Field(min_length=1, max_length=63, pattern=r"^[a-z0-9][a-z0-9-]*$")]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(Arguments):
    pass


class DatasetVersion(Arguments):
    revision: Revision = "HEAD"


class DatasetDiff(Arguments):
    old_revision: Revision
    new_revision: Revision = "HEAD"


class Runs(Arguments):
    experiment_ids: list[Identifier] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=5, ge=1, le=50)


class BestRun(Runs):
    metric: Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$", max_length=60)] = "f1"
    direction: Literal["maximize", "minimize"] = "maximize"


class Run(Arguments):
    run_id: Identifier


class CompareRuns(Arguments):
    run_ids: list[Identifier] = Field(min_length=2, max_length=10)


class Dag(Arguments):
    dag_id: Identifier = "training_pipeline"


class DagRun(Dag):
    run_id: Identifier


class TaskLog(DagRun):
    task_id: Identifier
    try_number: int = Field(default=1, ge=1, le=20)


class TriggerDag(Dag):
    c: float = Field(default=1.0, ge=0.001, le=100)
    minimum_f1: float = Field(default=0.90, ge=0, le=1)
    inject_validation_failure: bool = False


class Cluster(Arguments):
    namespace: Namespace | None = Field(default=None, description="Defaults to KUBERNETES_NAMESPACE")


class PodLog(Cluster):
    pod: Identifier
    tail_lines: int = Field(default=80, ge=1, le=200)


class Restart(Cluster):
    deployment: Identifier = "iris-api"


class ToolResult(BaseModel):
    evidence_id: str
    request_id: str
    tool: str
    source: str
    status: Literal["ok", "error", "confirmation_required"]
    observed_at: str
    duration_ms: float
    data: Any = None
    error: dict[str, str] | None = None


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    status: Literal["complete", "incomplete", "confirmation_required"]
    evidence: list[ToolResult] = Field(default_factory=list)
    model: str | None = None
