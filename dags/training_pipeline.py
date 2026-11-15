"""Airflow 3 TaskFlow DAG; user-supplied conf selects only bounded training parameters."""

import hashlib
import logging
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task

# Keep scientific/MLflow imports inside tasks: the scheduler repeatedly parses
# this module under a short import timeout, including on a cold WSL startup.


def context() -> tuple[Path, dict, str]:
    ctx = get_current_context()
    run_id = ctx["run_id"]
    directory = hashlib.sha256(run_id.encode()).hexdigest()[:20]
    conf = ctx["dag_run"].conf or {}
    return Path(__file__).resolve().parents[1] / ".runtime/training" / directory, conf, run_id


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=10),
        "execution_timeout": timedelta(minutes=5),
    },
    tags=["mlops", "assignment03"],
)
def training_pipeline():
    @task
    def validate_dataset():
        work, conf, _ = context()
        # Explicit reproducible failure injection; never evaluates arbitrary paths or code.
        if conf.get("inject_validation_failure", False):
            raise ValueError("Injected validation failure: dataset schema missing target column")
        from training import train as stages

        stages.validate(work)

    @task
    def preprocess_data():
        from training import train as stages

        work, _, _ = context()
        stages.preprocess(work)

    @task
    def train_model():
        from training import train as stages

        work, conf, _ = context()
        c = float(conf.get("c", 1.0))
        if not 0.001 <= c <= 100:
            raise ValueError("C must be between 0.001 and 100")
        stages.train(work, c)

    @task
    def evaluate_and_log():
        from training import train as stages

        work, _, run_id = context()
        logging.info("Evaluating training workflow %s", run_id)
        return stages.evaluate(work, run_id)["run_id"]

    @task
    def apply_quality_gate():
        from training import train as stages

        work, conf, _ = context()
        return stages.quality_gate(work, float(conf.get("minimum_f1", 0.9)))

    validate_dataset() >> preprocess_data() >> train_model() >> evaluate_and_log() >> apply_quality_gate()


training_pipeline()
