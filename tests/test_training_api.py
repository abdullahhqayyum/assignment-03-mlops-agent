"""Quality-gate and inference tests catch failures that would otherwise permit bad promotion."""

import json
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api.main import app
from scripts import promote_model
from training import train


def test_quality_gate_failure_marks_run_and_blocks_registration(tmp_path, monkeypatch):
    train.write_json(tmp_path / "metrics.json", {"run_id": "test-run", "metrics": {"f1": 0.4}})
    tags = []

    class Client:
        def set_tag(self, *args):
            tags.append(args)

    monkeypatch.setattr(train, "MlflowClient", lambda **kw: Client())
    monkeypatch.setattr(train.mlflow, "register_model", lambda *a, **kw: pytest.fail("Must not register"))
    with pytest.raises(ValueError, match="Quality gate failed"):
        train.quality_gate(tmp_path, 0.9)
    assert ("test-run", "quality_gate", "failed") in tags


def test_validation_rejects_bad_schema_before_training(tmp_path):
    source = tmp_path / "invalid.csv"
    source.write_text("wrong,target\n1,0\n")
    with pytest.raises(ValueError, match="schema"):
        train.validate(tmp_path, source)
    assert not (tmp_path / "validated.npy").exists()


def test_promotion_refreshes_gate_state_from_verified_run(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.joblib").write_bytes(b"model fixture")
    (artifacts / "metadata.json").write_text(
        json.dumps({"run_id": "passed-run", "dataset": {"quality_gate": "pending"}})
    )
    run = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(tags={"quality_gate": "passed", "quality_gate.minimum_f1": "0.9"}),
    )
    client = SimpleNamespace(
        get_run=lambda _: run,
        download_artifacts=lambda run_id, path, dest: str(artifacts / path.split("/")[-1]),
    )
    monkeypatch.setattr(promote_model, "ROOT", tmp_path)
    monkeypatch.setattr(promote_model, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(sys, "argv", ["promote_model.py", "--run-id", "passed-run"])
    promote_model.main()
    deployed = json.loads((tmp_path / "models/metadata.json").read_text())
    assert deployed["run_id"] == "passed-run"
    assert deployed["dataset"]["quality_gate"] == "passed"
    assert deployed["dataset"]["quality_gate.minimum_f1"] == "0.9"


def test_unavailable_model_fails_readiness_and_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/predict", json={"features": [5, 3, 1, 0.2]}).status_code == 503


def test_prediction_returns_model_provenance_and_validates_input(tmp_path, monkeypatch):
    model = DummyClassifier(strategy="constant", constant=0).fit(np.ones((2, 4)), [0, 0])
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"run_id": "recorded-run"}))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    with TestClient(app) as client:
        result = client.post("/predict", json={"features": [5, 3, 1, 0.2]})
        assert result.status_code == 200 and result.json()["run_id"] == "recorded-run"
        assert result.json()["class_name"] == "setosa"
        assert client.post("/predict", json={"features": [5, 3]}).status_code == 422
