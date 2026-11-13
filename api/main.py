"""Containerized Iris inference: health, readiness, provenance, and bounded prediction inputs."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
    app.state.model = None
    app.state.metadata = {}
    try:
        app.state.model = joblib.load(path)
        app.state.metadata = json.loads(path.with_name("metadata.json").read_text())
        LOG.info("Loaded model run_id=%s", app.state.metadata.get("run_id"))
    except (OSError, ValueError, KeyError):
        LOG.exception("Model is unavailable; readiness will fail")
    yield


app = FastAPI(title="Iris Inference API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[Annotated[float, Field(allow_inf_nan=False, gt=0)]] = Field(min_length=4, max_length=4)


@app.get("/healthz")
def health():
    return {"status": "alive", "version": os.environ.get("APP_VERSION", "1.0.0")}


@app.get("/readyz")
def ready():
    if app.state.model is None or not app.state.metadata.get("run_id"):
        raise HTTPException(503, "Model or provenance metadata unavailable")
    return {"status": "ready", "run_id": app.state.metadata["run_id"]}


@app.get("/version")
def version():
    return {"app_version": os.environ.get("APP_VERSION", "1.0.0"), **app.state.metadata}


@app.post("/predict")
def predict(request: PredictionRequest):
    ready()
    result = int(app.state.model.predict([request.features])[0])
    return {
        "class_id": result,
        "class_name": ["setosa", "versicolor", "virginica"][result],
        "run_id": app.state.metadata["run_id"],
    }
