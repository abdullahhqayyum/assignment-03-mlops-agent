"""Adapter tests use real SDK value shapes and mocked remote services."""

import json
from types import SimpleNamespace as Obj

import httpx
import pytest

from agent.config import Settings
from agent.logging import Redactor
from agent.registry import Registry, Tool
from agent.schemas import Cluster, DatasetVersion, TriggerDag
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DVCTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MLflowTools


def test_airflow_uses_authenticated_v2_api():
    observed = []

    def service(request):
        observed.append(request)
        if request.url.path == "/auth/token":
            assert json.loads(request.content)["username"] == "admin"
            return httpx.Response(200, json={"access_token": "private-bearer"})
        assert request.headers["Authorization"] == "Bearer private-bearer"
        return httpx.Response(
            200, json={"dags": [{"dag_id": "training_pipeline", "is_paused": False}], "total_entries": 1}
        )

    client = httpx.Client(base_url="http://airflow.test", transport=httpx.MockTransport(service))
    result = AirflowTools(Settings(), client).list_dags()
    assert result["dags"][0]["dag_id"] == "training_pipeline"
    assert observed[-1].url.path == "/api/v2/dags"


def test_airflow_dag_not_allowlisted_never_triggers():
    tools = AirflowTools(Settings(), client=Obj(post=lambda **kw: pytest.fail("No request allowed")))
    with pytest.raises(PermissionError):
        tools.trigger("dangerous_dag")


def test_training_trigger_does_not_run_without_confirmation():
    registry = Registry(
        Settings(),
        [
            Tool(
                "airflow_trigger",
                "trigger",
                TriggerDag,
                lambda **kwargs: pytest.fail("Must not trigger automatically"),
                True,
            )
        ],
    )
    result = registry.execute("airflow_trigger", {}, "req")
    assert result.status == "confirmation_required"
    assert result.data["executed"] is False


def test_airflow_http_failure_is_not_empty_success():
    client = httpx.Client(
        base_url="http://airflow.test", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(httpx.HTTPStatusError):
        AirflowTools(Settings(), client).list_dags()


def fake_run(run_id, metric, status="FINISHED"):
    return Obj(
        info=Obj(run_id=run_id, experiment_id="1", status=status, start_time=1700000000000),
        data=Obj(metrics={"loss": metric}, params={}, tags={"quality_gate": "passed"}),
    )


def test_model_selection_honors_minimize_and_excludes_nonfinite_or_running():
    runs = [
        fake_run("good", 0.2),
        fake_run("worse", 0.8),
        fake_run("invalid", float("nan")),
        fake_run("unfinished", 0.1, "RUNNING"),
    ]
    client = Obj(search_runs=lambda *args, **kw: runs)
    result = MLflowTools(Settings(), client).best_run(["1"], metric="loss", direction="minimize")
    assert result["best"]["run_id"] == "good"
    assert result["freshness_policy_exceeded"] is None
    assert "does not establish production readiness" in result["caution"]


def test_no_matching_metric_is_reported_as_no_candidate():
    client = Obj(search_runs=lambda *args, **kw: [fake_run("no-f1", 0.2)])
    assert MLflowTools(Settings(), client).best_run(["1"])["best"] is None


def test_kubernetes_namespace_guard_precedes_api_access():
    tools = KubernetesTools(Settings())
    with pytest.raises(PermissionError):
        tools.workload_status("default")


def test_dvc_shell_injection_revision_is_invalid():
    registry = Registry(
        Settings(),
        [
            Tool(
                "dvc_dataset_version",
                "version",
                DatasetVersion,
                lambda **kwargs: pytest.fail("Must not execute"),
            )
        ],
    )
    result = registry.execute("dvc_dataset_version", {"revision": "HEAD; rm -rf /"}, "req")
    assert result.error["code"] == "invalid_request"


def test_dvc_reads_historical_git_pointer_not_current_workspace(monkeypatch):
    tools = DVCTools(Settings())
    commands = []

    def command(*args):
        commands.append(args)
        return "abc123" if args[1] == "rev-parse" else "outs:\n- md5: recorded\n  path: iris.csv\n"

    monkeypatch.setattr(tools, "command", command)
    result = tools.dataset_version("dataset-v1")
    assert result["outputs"][0]["md5"] == "recorded"
    assert commands[-1] == ("git", "show", "abc123:data/iris.csv.dvc")


def test_redaction_covers_nested_values_and_configured_secret():
    value = {
        "password": "hidden",
        "logs": "failure credential-123 api_key=abcd",
        "nested": [{"token": "hidden"}],
    }
    cleaned = Redactor(["credential-123"]).clean(value)
    assert "hidden" not in json.dumps(cleaned)
    assert "credential-123" not in json.dumps(cleaned)
    assert "abcd" not in json.dumps(cleaned)


def test_running_version_reads_raw_service_response():
    closed = []
    response = Obj(
        data=b'{"run_id":"actual-running-model","app_version":"1.0.0"}', close=lambda: closed.append(True)
    )
    core = Obj(connect_get_namespaced_service_proxy_with_path=lambda *args, **kwargs: response)
    result = KubernetesTools(Settings(), core=core, apps=Obj()).app_version()
    assert result["application"]["run_id"] == "actual-running-model"
    assert closed == [True]


def test_unknown_bearer_and_url_credentials_are_scrubbed():
    text = "Authorization: Bearer opaque123 https://user:private-password@example.test"
    result = Redactor().clean(text)
    assert "opaque123" not in result and "private-password" not in result


@pytest.mark.parametrize("arguments", [{}, {"namespace": None}])
def test_omitted_namespace_uses_configuration_and_still_rejects_other_namespaces(arguments):
    registry = Registry(
        Settings(namespace="team-demo"),
        [Tool("kubernetes_workloads", "status", Cluster, lambda namespace: {"namespace": namespace})],
    )
    result = registry.execute("kubernetes_workloads", arguments, "configured-namespace")
    assert result.status == "ok"
    assert result.data["namespace"] == "team-demo"
    denied = registry.execute("kubernetes_workloads", {"namespace": "mlops-agent"}, "wrong-namespace")
    assert denied.error["code"] == "unauthorized"


def test_direct_adapter_uses_configured_namespace():
    observed = []

    def proxy(service, namespace, path, **kwargs):
        observed.append(namespace)
        return Obj(data=b'{"run_id":"actual-model"}', close=lambda: None)

    core = Obj(connect_get_namespaced_service_proxy_with_path=proxy)
    result = KubernetesTools(Settings(namespace="team-demo"), core=core, apps=Obj()).app_version()
    assert observed == ["team-demo"]
    assert result["namespace"] == "team-demo"


@pytest.mark.parametrize("kubeconfig", [None, "/custom/cluster-config"])
def test_live_prediction_uses_configured_cluster_without_a_fixed_local_port(kubeconfig, monkeypatch):
    from scripts import verify_live

    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return Obj(stdout='{"class_name":"setosa","run_id":"actual-model"}')

    monkeypatch.setattr(verify_live.subprocess, "run", run)
    sample = {"features": [5.1, 3.5, 1.4, 0.2]}
    result = verify_live.predict_through_service(
        Settings(namespace="team-demo", kubeconfig=kubeconfig), sample
    )
    command, kwargs = observed[0]
    assert result["run_id"] == "actual-model"
    assert "/api/v1/namespaces/team-demo/services/iris-api:80/proxy/predict" in command
    assert json.loads(kwargs["input"]) == sample
    assert ("--kubeconfig" in command) == bool(kubeconfig)
    assert "port-forward" not in command and None not in command


def test_configure_uses_custom_airflow_username_without_printing_credentials(tmp_path, monkeypatch, capsys):
    import sys

    from scripts import configure

    (tmp_path / ".env.example").write_text(
        "AIRFLOW_USERNAME=demo-operator\nAIRFLOW_PASSWORD=local-test-value\n"
    )
    monkeypatch.setattr(configure, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["configure.py"])
    configure.main()
    passwords = json.loads((tmp_path / ".runtime/airflow-secrets/passwords.json").read_text())
    assert passwords == {"demo-operator": "local-test-value"}
    assert "local-test-value" not in capsys.readouterr().out


def test_configure_rejects_airflow_username_role_injection(tmp_path, monkeypatch):
    import sys

    from scripts import configure

    (tmp_path / ".env.example").write_text("AIRFLOW_USERNAME=operator:admin,other:admin\n")
    monkeypatch.setattr(configure, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["configure.py"])
    with pytest.raises(SystemExit, match="AIRFLOW_USERNAME"):
        configure.main()
    assert not (tmp_path / ".env").exists()
