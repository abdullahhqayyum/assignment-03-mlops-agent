# Architecture

Natural-language request -> FastAPI/CLI -> LLM -> validated tool dispatcher ->
DVC / MLflow / Airflow 3 / Kubernetes -> timestamped evidence -> response.

The LLM selects structured tools. The dispatcher validates arguments, enforces
namespace and DAG allowlists, records tool status and timing, and returns failures
as failures. Cluster mutations require a separate user confirmation that cannot
be supplied as a model-generated tool argument.

Training stages: validate dataset -> preprocess -> train -> evaluate/log to
MLflow -> quality gate. Deployment is an explicit separate operation; neither
model ranking nor the integrated diagnostic request automatically retrains or deploys.

Real-service evidence and mocked CI results will be documented separately.
