"""Offline control-flow tests use scripted LLM messages, never a real API key."""

from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage

from agent.config import Settings
from agent.orchestrator import Agent
from agent.registry import Registry, Tool
from agent.schemas import Empty, Restart


def message(tool=None, arguments="{}", content=None):
    return ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": tool, "arguments": arguments}}]
        if tool
        else None,
    )


class ScriptedLLM:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.messages))])


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr("agent.orchestrator.time.sleep", lambda seconds: None)


def test_llm_selects_executes_and_observes_tool():
    settings = Settings()
    registry = Registry(
        settings, [Tool("mlflow_list_experiments", "list", Empty, lambda: {"experiments": [{"id": "1"}]})]
    )
    llm = ScriptedLLM([message("mlflow_list_experiments"), message(content="Experiment 1 was returned.")])
    result = Agent(settings, registry, llm).run("List experiments")
    assert result.status == "complete"
    assert result.evidence[0].data["experiments"][0]["id"] == "1"
    assert llm.calls[1]["messages"][-1]["role"] == "tool"


def test_retraining_without_policy_or_degradation_requires_uncertainty():
    settings = Settings()
    registry = Registry(
        settings,
        [
            Tool(
                "mlflow_best_run",
                "best",
                Empty,
                lambda: {
                    "best": {"tags": {"quality_gate": "passed"}},
                    "freshness_policy_exceeded": None,
                },
            )
        ],
    )
    llm = ScriptedLLM(
        [
            message("mlflow_best_run"),
            message(content="Retraining is not necessary because the gate passed."),
            message(content="Retraining necessity: unknown. No production degradation data was supplied."),
        ]
    )
    result = Agent(settings, registry, llm).run("Is retraining necessary?")
    assert result.status == "complete"
    assert "Retraining necessity: unknown" in result.answer
    assert len(llm.calls) == 3


def test_repeated_unsupported_retraining_conclusion_is_incomplete():
    settings = Settings()
    registry = Registry(settings, [Tool("mlflow_best_run", "best", Empty, lambda: {"best": None})])
    llm = ScriptedLLM(
        [
            message("mlflow_best_run"),
            message(content="No retraining is needed."),
            message(content="Retraining is definitely unnecessary."),
        ]
    )
    result = Agent(settings, registry, llm).run("Is retraining necessary?")
    assert result.status == "incomplete"
    assert "Retraining necessity: unknown" in result.answer
    assert "definitely" not in result.answer


def test_unavailable_tool_cannot_be_replaced_with_invented_success():
    def unavailable():
        raise ConnectionError("Tracking service unavailable")

    settings = Settings()
    llm = ScriptedLLM(
        [message("mlflow_list_experiments"), message(content="All models are healthy, F1 is 0.99")]
    )
    registry = Registry(settings, [Tool("mlflow_list_experiments", "list", Empty, unavailable)])
    result = Agent(settings, registry, llm).run("How are my models?")
    assert result.status == "incomplete"
    assert "unknown" in result.answer
    assert "0.99" not in result.answer
    assert len(llm.calls) == 1
    assert result.evidence[0].error["code"] == "service_unavailable"


def test_multi_tool_followup_receives_actual_previous_evidence():
    settings = Settings()
    registry = Registry(
        settings,
        [
            Tool("mlflow_best_run", "best", Empty, lambda: {"dataset_commit": "abc123"}),
            Tool("dvc_dataset_version", "data", Empty, lambda: {"hash": "recorded_hash"}),
        ],
    )
    llm = ScriptedLLM(
        [
            message("mlflow_best_run"),
            message("dvc_dataset_version"),
            message(content="Selected run uses recorded_hash"),
        ]
    )
    result = Agent(settings, registry, llm).run("Find model and dataset")
    assert [e.source for e in result.evidence] == ["mlflow", "dvc"]
    assert result.evidence[1].data["hash"] == "recorded_hash"


def test_malformed_model_arguments_are_rejected():
    settings = Settings()
    registry = Registry(
        settings, [Tool("dvc_status", "status", Empty, lambda: pytest.fail("Must not execute"))]
    )
    result = Agent(settings, registry, ScriptedLLM([message("dvc_status", "not json")])).run("Inspect data")
    assert result.evidence[0].error["code"] == "invalid_request"


def test_model_cannot_supply_confirmation_as_argument():
    settings = Settings()
    registry = Registry(
        settings,
        [
            Tool(
                "kubernetes_restart",
                "restart",
                Restart,
                lambda **kwargs: pytest.fail("Must not execute"),
                True,
            )
        ],
    )
    result = registry.execute("kubernetes_restart", {"namespace": "mlops-agent", "confirmed": True}, "req")
    assert result.error["code"] == "invalid_request"


def test_restart_requires_separate_one_use_human_confirmation():
    calls = []
    settings = Settings()
    registry = Registry(
        settings,
        [
            Tool(
                "kubernetes_restart",
                "restart",
                Restart,
                lambda **kwargs: calls.append(kwargs) or {"requested": True},
                True,
            )
        ],
    )
    result = registry.execute("kubernetes_restart", {}, "req")
    assert result.status == "confirmation_required" and calls == []
    confirmed = registry.confirm(result.data["confirmation_id"], True)
    assert confirmed.status == "ok" and len(calls) == 1
    with pytest.raises(ValueError):
        registry.confirm(result.data["confirmation_id"], True)


def test_unauthorized_namespace_rejected_before_modification():
    settings = Settings()
    registry = Registry(
        settings,
        [
            Tool(
                "kubernetes_restart",
                "restart",
                Restart,
                lambda **kwargs: pytest.fail("Must not execute"),
                True,
            )
        ],
    )
    result = registry.execute("kubernetes_restart", {"namespace": "kube-system"}, "req")
    assert result.error["code"] == "unauthorized"
    assert registry.pending == {}


def test_declined_and_expired_confirmation_never_execute():
    registry = Registry(
        Settings(),
        [
            Tool(
                "kubernetes_restart",
                "restart",
                Restart,
                lambda **kwargs: pytest.fail("Must not execute"),
                True,
            )
        ],
    )
    result = registry.execute("kubernetes_restart", {}, "req")
    with pytest.raises(PermissionError):
        registry.confirm(result.data["confirmation_id"], False)
    result = registry.execute("kubernetes_restart", {}, "req")
    key = result.data["confirmation_id"]
    name, values, req, _ = registry.pending[key]
    registry.pending[key] = (name, values, req, 0)
    with pytest.raises(ValueError):
        registry.confirm(key, True)


def test_tool_timeout_recorded_and_logs_redacted(caplog):
    def timeout():
        raise TimeoutError("password=hunter2 sk-or-v1-example123")

    caplog.set_level("INFO")
    registry = Registry(Settings(), [Tool("dvc_status", "status", Empty, timeout)])
    result = registry.execute("dvc_status", {}, "request-123")
    assert result.error["code"] == "timeout"
    assert "hunter2" not in result.model_dump_json() + caplog.text
    assert "sk-or-v1-example123" not in caplog.text
    assert '"duration_ms"' in caplog.text and "request-123" in caplog.text


def test_agent_does_not_claim_state_without_tools():
    result = Agent(
        Settings(), Registry(Settings(), []), ScriptedLLM([message(content="Everything healthy")])
    ).run("Status?")
    assert result.status == "incomplete"
    assert "Everything healthy" not in result.answer
