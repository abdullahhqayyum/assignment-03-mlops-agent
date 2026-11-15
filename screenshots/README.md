# Demonstration screenshots

Captured on **15 November 2026** against the real local Ubuntu WSL stack.
The native MLflow and Airflow images are direct browser screenshots. The other
images show a clearly labelled evidence viewer containing actual command output,
HTTP responses, or previously saved live tool responses; they are not service UI
mockups. Credentials and authentication screens are excluded.

| Screenshot | What it demonstrates | Capture/source |
|---|---|---|
| [01 · MLflow experiments](01-mlflow-runs.png) | Eight recorded runs, including the latest Airflow training runs and the C=10 candidate | Live MLflow UI |
| [01b · Selected model](01b-mlflow-best-model.png) | F1, accuracy, parameters, dataset provenance tags, passed quality gate and registered model | Live MLflow run UI |
| [02 · Airflow run history](02-airflow-pipeline.png) | Successful recovery and both deliberately failed runs | Live Airflow UI |
| [02b · Successful workflow](02b-airflow-success.png) | Five successful task instances in the actual recovery run | Live Airflow run UI |
| [03 · DVC provenance](03-dvc-provenance.png) | Git revision, historical dataset pointers, clean data status and matching restored dataset hash | Fresh Git/DVC commands and [restoration export](../docs/evidence/live-recheck-dataset-2026-11-15.json) |
| [04 · Kubernetes health](04-kubernetes-health.png) | Two ready replicas, ClusterIP Service, ready endpoints and real health-check logs | Fresh kubectl output |
| [05 · Integrated agent](05-integrated-agent.png) | Actual four-system tool calls, model/data/deployment correlation and retraining uncertainty | [Recorded live LLM conversation](../docs/evidence/live-recheck-2026-11-15.json) |
| [06 · Prediction](06-live-prediction.png) | HTTP 200 readiness/version checks and a real `setosa` prediction with deployed run ID | Fresh HTTP requests through the Kubernetes Service |
| [07 · Quality gate failure](07-quality-gate-failure.png) | F1 below the required threshold, two gate attempts, failed task log and successful recovery reference | Saved real Airflow API responses |
| [08 · Validation failure](08-validation-failure.png) | Validation retried twice; all downstream tasks blocked; actual failure log | Saved real Airflow API responses |

[Raw screenshot command and HTTP captures](../docs/evidence/screenshot-capture-2026-11-15.json)
preserve the source data for screenshots 03, 04 and 06. The
[complete live verification](../docs/evidence/live-recheck-2026-11-15.json)
contains task states, logs, the LLM response and observation timestamps for 05,
07 and 08. Log excerpts are labelled in the images; full available diagnostics
remain in the JSON exports.

Some native service tables abbreviate long identifiers. Use the linked exports
for exact values. Pod restart counts include earlier WSL shutdowns; both replicas
were ready at capture time. These images document observations at capture time,
not a permanently hosted deployment.

Open the PNG files at their original size for readable text. See the
[technical report](../docs/report.md) and [local run instructions](../docs/RUN-LOCALLY.md).
[The capture manifest](manifest.json) records file timestamps, sizes and SHA-256 hashes.
