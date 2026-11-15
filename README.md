# Assignment 03: MLOps AI Agent

[![CI](https://github.com/abdullahhqayyum/assignment-03-mlops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullahhqayyum/assignment-03-mlops-agent/actions/workflows/ci.yml)

An LLM selects structured tools to inspect **DVC**, **MLflow**, **Airflow 3** and
**Kubernetes**. Answers include actual tool responses, timestamps and request IDs.
The project trains an Iris classifier and serves predictions with FastAPI.

For Windows/Ubuntu WSL startup and interaction, see
[local run instructions](docs/RUN-LOCALLY.md).

See [architecture](architecture.md), [technical report](docs/report.md), and
[exported live evidence](docs/evidence/). CI mocks external services; live evidence
is collected separately against running services and a real LLM.
The [screenshot index](screenshots/README.md) provides visual demonstration evidence.

## 1. Environment

Requires Python **3.11-3.13**, Git, Docker with Linux containers and Compose,
`kubectl`, [kind v0.30.0](https://github.com/kubernetes-sigs/kind/releases/tag/v0.30.0),
and an OpenRouter key. Allocate at least 6 GB RAM to Docker; allow about 12 GB free
disk space for the environment, images, unpacking and build layers.

**Ubuntu WSL:** Docker Engine can run directly in Ubuntu with Docker Desktop off.
Run `sudo systemctl start docker`, then perform the setup in the Ubuntu terminal.
Clone into the Linux home directory to avoid cross-system ownership and I/O issues.
Create a separate Linux environment (`python3 -m venv .venv-wsl` and
`source .venv-wsl/bin/activate`); a Windows `.venv` cannot be reused on Linux.
If reusing a Windows `.env`, export `AIRFLOW_UID=$(id -u)` and
`KUBECONFIG="$PWD/.runtime/kubeconfig"` in Ubuntu to override Windows-specific values.

```bash
git clone https://github.com/abdullahhqayyum/assignment-03-mlops-agent.git
cd assignment-03-mlops-agent
python -m venv .venv
```

Activate with `source .venv/bin/activate` (Linux/macOS) or
`.venv\Scripts\Activate.ps1` (PowerShell). Windows users can create the environment
with `py` if `python` is unavailable. Without activation, use
`.venv\Scripts\python.exe` or `.venv/bin/python` in the commands below.

```bash
python -m pip install -r requirements.txt
python scripts/configure.py
```

Set `OPENROUTER_API_KEY` in the generated **ignored** `.env`. To reuse an existing
key: `python scripts/configure.py --reuse-env /path/to/previous/.env`.
The script preserves existing settings and generates local Airflow/Postgres/JWT/
Fernet credentials without printing them. Airflow runs in its own pinned Linux image.

| Variable | Default/purpose |
|---|---|
| `OPENROUTER_API_KEY` | Required for live agent conversations |
| `LLM_MODEL` | `openrouter/free`; no silent fallback to a paid model |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `AIRFLOW_URL` | `http://localhost:8080` |
| `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | `admin` and a generated local password |
| `APPROVED_DAGS` | `training_pipeline` |
| `KUBERNETES_NAMESPACE` | `mlops-agent` |
| `KUBECONFIG` | Absolute `.runtime/kubeconfig` path, set by configure.py |
| `TOOL_TIMEOUT_SECONDS` | 15 per subprocess/HTTP/SDK request |
| `LLM_TIMEOUT_SECONDS`, `MAX_AGENT_STEPS` | 90 per API request; 16 model turns |
| `MAX_MODEL_AGE_DAYS` | Unset: no invented freshness policy |

Exported shell variables take precedence. `openrouter/free` can select different
models on different calls. Configure a current tool-capable free slug to pin a
model; see [OpenRouter routing](https://openrouter.ai/docs/guides/routing/routers/free-router).
Free model availability and account quotas can change.

Omitting a Kubernetes tool's namespace uses `KUBERNETES_NAMESPACE`; explicitly
requesting any other namespace is rejected. Match the namespace in the manifests
when changing this setting. The Iris schema and `iris-api` application identifiers
are the supplied demo's design and permission boundary; operational run IDs,
hashes, metrics and health always come from tool responses.

## 2. DVC data and historical reproduction

```bash
python scripts/bootstrap_data.py
python -m dvc data status --json
python scripts/reproduce_dataset.py --revision dataset-v1
```

The bootstrap script rebuilds both deterministic versions from scikit-learn's
bundled Iris data, populates the cache/local remote and restores the committed
pointer. No dataset download is needed. `dataset-v1` has 120 rows; `dataset-v2`
has 150. The local remote `.runtime/dvc-remote` is not committed. Production use
would replace it with a restricted durable remote.

The reproduction script pulls an old version into a temporary detached Git
worktree and checks its MD5 against the committed pointer, leaving the main
worktree's data unchanged. The agent explains historical checkout commands;
it never silently replaces working data.

## 3. MLflow and Airflow 3

```bash
docker compose up -d --build
docker compose ps
```

MLflow: <http://localhost:5000>. Airflow: <http://localhost:8080>, user `admin`,
password from your local `.env`. Both ports bind to loopback. Airflow's development
standalone launcher runs the API server, scheduler and DAG processor using
PostgreSQL and LocalExecutor. The adapter authenticates with JWT and uses `/api/v2`.
Wait until `training_pipeline` is visible before triggering a run.

To train without Airflow, while MLflow is running:

```bash
python -m training.train all --c 1.0
```

Or run `python -m dvc repro` for the same five stages. The raw dataset must match
its DVC hash and the pointer must match HEAD: commit a deliberate new dataset
version before training. MLflow records Git commit, dataset hash/path, parameters,
macro F1, accuracy, model artifacts, workflow ID and quality-gate outcome.
The gate registers passing candidates; deployment is an explicit separate step.

## 4. Kubernetes inference

Choose a real FINISHED MLflow run with `quality_gate=passed`:

```bash
python scripts/promote_model.py --run-id YOUR_RUN_ID
docker build -t mlops-iris:1.0.0 .
kind create cluster --name assignment03 --kubeconfig .runtime/kubeconfig --image kindest/node:v1.34.0 --wait 180s
kind load docker-image mlops-iris:1.0.0 --name assignment03
kubectl --kubeconfig .runtime/kubeconfig apply -f kubernetes/namespace.yaml
kubectl --kubeconfig .runtime/kubeconfig apply -f kubernetes/
kubectl --kubeconfig .runtime/kubeconfig -n mlops-agent rollout status deployment/iris-api --timeout=180s
kubectl --kubeconfig .runtime/kubeconfig -n mlops-agent port-forward service/iris-api 8000:80
```

In another terminal, check <http://localhost:8000/readyz> and
<http://localhost:8000/version>. Prediction in PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType application/json -Body '{"features":[5.1,3.5,1.4,0.2]}'
```

On Linux/macOS:

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Manifests specify **two replicas**, ClusterIP Service, probes, resource requests/
limits and ConfigMap. `/version` returns the running model's actual MLflow and
dataset references. Missing model/provenance fails readiness (503) independently
of liveness. Inference Pods run as non-root and do not mount service-account tokens.

`kubernetes/rbac.yaml` supplies an optional namespace-scoped read-only identity.
Local demonstrations use the isolated kind kubeconfig. Read-only credentials
cannot restart workloads even after application confirmation; explicitly grant
only the required Deployment patch permission if an operator needs that operation.

## 5. Agent CLI and HTTP interface

```bash
python -m agent.main "Compare the last five training runs and recommend the best candidate based on F1."
python -m agent.main "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary." --output .runtime/integrated.json
python -m agent.main "Trigger the model training pipeline." --interactive
```

The last command shows exact arguments and requires the human to type `YES`.
The same rule protects Kubernetes restarts. Confirmation grants expire after five
minutes and work once. Namespace/DAG allowlists are checked before execution;
model-generated arguments cannot provide a confirmation flag.

```bash
python -m uvicorn agent.server:app --host 127.0.0.1 --port 8001
```

At <http://localhost:8001/docs>, `POST /chat` takes `{"message":"..."}` and
`POST /confirm` takes `{"confirmation_id":"...","approved":true}`.
Confirmation IDs come from pending tool results and must be confirmed in the same
running process. The LLM has no confirmation tool. This development interface has
no multi-user authentication: keep it bound to loopback. Requests are independent;
include identifiers explicitly in follow-up questions.

Responses include status (`complete`, `incomplete`, `confirmation_required`) and
full evidence. Tool failure means unknown, never healthy. Diagnostic questions do
not start retraining. Without a supplied freshness policy, failed gate or model
degradation evidence, retraining necessity may remain unknown.

## 6. Testing and evidence

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI runs lint, formatting, tests and DAG syntax compilation on pull requests and
main pushes for Python 3.11 and 3.13. Tests need no keys or services. They cover
selection/execution, timeouts, invalid inputs, service failures, missing evidence,
namespace/confirmation rules, quality gates and inference readiness. Live Airflow
import/execution is checked separately in its container.

Recheck an existing running stack without rebuilding or replacing the deployment:

```bash
python scripts/verify_live.py --approve-training --with-llm --output .runtime/live-verification.json
```

This uses real services: historical DVC restoration, MLflow records, Kubernetes
health/logs/endpoints and an actual prediction. `--approve-training` authorizes
two deliberately failed DAG runs (quality gate and validation), followed by a
successful recovery run. Omit that flag for read-only service checks.
`--with-llm` additionally verifies the mandatory four-service conversation using
the configured API key. The script exits unsuccessfully if a required check fails
and preserves the observations collected so far. Allow several minutes on WSL.

Capture another real conversation (consumes LLM quota):

```bash
python scripts/capture_conversation.py --name provenance --request "Show dataset-v1 provenance and how to reproduce it." --output .runtime/conversation.json
```

Add `--interactive` to confirm a training demonstration. The script then waits for
that run's terminal state and asks the agent to inspect the outcome. The technical
report links the five required conversations and two failure investigations.
Never commit `.env`, kubeconfig, service tokens, `.runtime` or private artifacts.

The optional **Dedicated live demonstration** Actions workflow provisions a real
stack on a disposable Linux runner. It runs only by manual dispatch with explicit
approval for training and namespace-scoped fault injection. Enable `with_llm` and
configure an encrypted `OPENROUTER_API_KEY` repository secret to capture real
agent conversations as well. It uploads exported evidence as an artifact; normal
pull-request CI continues to use mocks and no credentials.

## 7. Shutdown

Stop port-forwarding with Ctrl+C. `docker compose stop` retains service data.
`kind delete cluster --name assignment03` removes only the demo cluster; reload
the inference image after recreating it.
