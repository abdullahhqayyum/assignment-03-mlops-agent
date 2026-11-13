"""Five independently callable stages shared by DVC, Airflow and the local CLI."""

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.data import FEATURES

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate(work: Path, source: Path | None = None) -> dict:
    source = source or ROOT / "data/iris.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [*FEATURES, "target"]:
            raise ValueError("Dataset schema must contain four Iris features and target")
        rows = list(reader)
    matrix = np.array([[float(row[key]) for key in [*FEATURES, "target"]] for row in rows])
    if len(rows) < 30 or not np.isfinite(matrix).all():
        raise ValueError("Dataset must contain >=30 rows and only finite numeric values")
    if set(matrix[:, 4]) != {0, 1, 2} or (matrix[:, :4] <= 0).any():
        raise ValueError("Dataset requires three target classes and positive feature values")
    pointer = yaml.safe_load((ROOT / "data/iris.csv.dvc").read_text())["outs"][0]
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    if digest != pointer["md5"]:
        raise ValueError("Dataset content differs from recorded DVC hash; version it before training")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    committed = subprocess.check_output(
        ["git", "show", f"{git_commit}:data/iris.csv.dvc"], cwd=ROOT, text=True, timeout=10
    )
    if yaml.safe_load(committed)["outs"][0]["md5"] != digest:
        raise ValueError("DVC reference must be committed before training")
    metadata = {
        "dataset.git_commit": git_commit,
        "dataset.dvc_hash": digest,
        "dataset.path": "data/iris.csv",
        "dataset.rows": str(len(rows)),
    }
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / "validated.npy", matrix)
    write_json(work / "provenance.json", metadata)
    LOG.info("Validated %s rows at dataset hash %s", len(rows), digest)
    return metadata


def preprocess(work: Path) -> None:
    matrix = np.load(work / "validated.npy")
    x_train, x_test, y_train, y_test = train_test_split(
        matrix[:, :4], matrix[:, 4].astype(int), test_size=0.25, random_state=42, stratify=matrix[:, 4]
    )
    np.savez(work / "split.npz", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    LOG.info("Prepared fixed stratified split; scaling will fit on training data only")


def train(work: Path, c: float = 1.0) -> None:
    split = np.load(work / "split.npz")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300, random_state=42))
    model.fit(split["x_train"], split["y_train"])
    joblib.dump(model, work / "model.joblib")
    write_json(work / "parameters.json", {"C": c, "random_state": 42, "test_size": 0.25})
    LOG.info("Trained scaled logistic regression C=%s", c)


def evaluate(work: Path, workflow_run: str = "local") -> dict:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("iris-classification")
    split = np.load(work / "split.npz")
    model = joblib.load(work / "model.joblib")
    prediction = model.predict(split["x_test"])
    metrics = {
        "f1": float(f1_score(split["y_test"], prediction, average="macro")),
        "accuracy": float(accuracy_score(split["y_test"], prediction)),
    }
    tags = read_json(work / "provenance.json") | {"airflow.run_id": workflow_run, "quality_gate": "pending"}
    with mlflow.start_run(tags=tags, run_name=workflow_run) as run:
        mlflow.log_params(read_json(work / "parameters.json"))
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(model, name="iris-classifier")
        metadata = {
            "run_id": run.info.run_id,
            "dataset": tags,
            "metrics": metrics,
            "model_uri": info.model_uri,
        }
        write_json(work / "metadata.json", metadata)
        mlflow.log_artifact(str(work / "model.joblib"), "inference")
        mlflow.log_artifact(str(work / "metadata.json"), "inference")
        mlflow.log_artifact(str(work / "provenance.json"), "provenance")
        write_json(work / "metrics.json", metadata)
    LOG.info("MLflow run %s metrics=%s", metadata["run_id"], metrics)
    return metadata


def quality_gate(work: Path, minimum_f1: float = 0.90) -> dict:
    if not 0 <= minimum_f1 <= 1:
        raise ValueError("minimum_f1 must be between 0 and 1")
    result = read_json(work / "metrics.json")
    client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    passed = result["metrics"]["f1"] >= minimum_f1
    client.set_tag(result["run_id"], "quality_gate", "passed" if passed else "failed")
    client.set_tag(result["run_id"], "quality_gate.minimum_f1", str(minimum_f1))
    if not passed:
        raise ValueError(f"Quality gate failed: F1={result['metrics']['f1']:.4f} < {minimum_f1}")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    version = mlflow.register_model(result["model_uri"], "iris-classifier")
    client.set_model_version_tag(
        "iris-classifier", version.version, "dataset.dvc_hash", result["dataset"]["dataset.dvc_hash"]
    )
    LOG.info(
        "Quality gate passed; registered candidate version %s (no automatic deployment)", version.version
    )
    return {"run_id": result["run_id"], "registered_version": version.version, "passed": True}


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "preprocess", "train", "evaluate", "gate", "all"])
    parser.add_argument("--work", type=Path, default=ROOT / ".runtime/training/local")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--minimum-f1", type=float, default=0.90)
    args = parser.parse_args()
    stages = {
        "validate": lambda: validate(args.work),
        "preprocess": lambda: preprocess(args.work),
        "train": lambda: train(args.work, args.c),
        "evaluate": lambda: evaluate(args.work),
        "gate": lambda: quality_gate(args.work, args.minimum_f1),
    }
    for name in stages if args.stage == "all" else [args.stage]:
        stages[name]()


if __name__ == "__main__":
    main()
