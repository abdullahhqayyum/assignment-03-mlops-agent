"""Explicitly materialize a quality-approved MLflow run for the next inference image build."""

import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    run = client.get_run(args.run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("quality_gate") != "passed":
        raise SystemExit("Only a FINISHED run with a passed quality gate can be prepared for deployment")
    destination = ROOT / "models"
    destination.mkdir(exist_ok=True)
    staging = ROOT / ".runtime/downloads"
    staging.mkdir(parents=True, exist_ok=True)
    for name in ("model.joblib", "metadata.json"):
        downloaded = client.download_artifacts(args.run_id, "inference/" + name, str(staging))
        shutil.copy2(downloaded, destination / name)
    metadata = json.loads((destination / "metadata.json").read_text())
    if metadata["run_id"] != args.run_id:
        raise SystemExit("Downloaded model provenance does not match requested run")
    print("Prepared model run", args.run_id, "for an explicit image build; cluster unchanged.")


if __name__ == "__main__":
    main()
