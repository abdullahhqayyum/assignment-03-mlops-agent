# Run locally with Ubuntu WSL

Follow the [main README](../README.md) for a fresh installation, including Python
dependencies, generated credentials, DVC bootstrap, training and Kubernetes image
deployment. This guide explains how to use or resume that installation.

Keep an Ubuntu terminal open while using a Docker Engine installed inside WSL;
background system services alone do not keep the distribution running. If using
Docker Desktop instead, enable its WSL integration and use that Docker daemon
consistently for Compose and kind.

From PowerShell, open your installed distribution:

```powershell
wsl -d Ubuntu
```

Inside Ubuntu, enter your Linux clone and activate the Linux virtual environment
you created during setup. For a clone and environment created as in the README:

```bash
cd ~/assignment-03-mlops-agent
source .venv-wsl/bin/activate
export KUBECONFIG="$PWD/.runtime/kubeconfig"
export TOOL_TIMEOUT_SECONDS=60
export KUBERNETES_NAMESPACE=mlops-agent
```

Use your actual clone/environment locations if you chose different ones. Do not
reuse a Windows virtual environment inside Linux. `kind` and `kubectl` must be on
your PATH; no executable is bundled in this repository.

For a native Docker Engine, start it with `sudo systemctl start docker`. Resume
the existing demonstration services and cluster:

```bash
docker compose up -d
docker start assignment03-control-plane
kubectl -n "$KUBERNETES_NAMESPACE" rollout status deployment/iris-api --timeout=180s
```

`assignment03-control-plane` is the node created by the README's
`kind create cluster --name assignment03` command. If you chose another cluster
name, use its node name from `docker ps -a`. A missing cluster must be created and
loaded with the inference image as documented in the README.

Cold starts can take a few minutes. Wait for Airflow to display
`training_pipeline` before triggering training. If kind reports a transient cgroup
initialization error during WSL startup, let Docker finish starting and retry the
node start command.

Start the conversational API:

```bash
python -m uvicorn agent.server:app --host 127.0.0.1 --port 8001
```

Open [agent API documentation](http://localhost:8001/docs) in the Windows browser.
Windows localhost forwarding reaches the server inside WSL. Use `POST /chat`:

```json
{"message":"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}
```

The development API has no user authentication; keep it bound to loopback. For
CLI interaction, open another configured Ubuntu terminal in the repository:

```bash
python -m agent.main "Compare the last five training runs based on F1."
python -m agent.main "Trigger the model training pipeline." --interactive
```

The second command requests separate confirmation for the displayed operation.
An accepted trigger means queued; inspect the returned run ID for the final result.

To access inference directly, use another configured terminal:

```bash
kubectl -n "$KUBERNETES_NAMESPACE" port-forward service/iris-api 8000:80
```

Then use PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/readyz
Invoke-RestMethod http://localhost:8000/version
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType application/json -Body '{"features":[5.1,3.5,1.4,0.2]}'
```

Other interfaces: [MLflow](http://localhost:5000) and
[Airflow](http://localhost:8080). Airflow credentials come from the Linux clone's
ignored `.env`, generated during setup. Service endpoints, model choice,
credentials, namespace, kubeconfig and timeouts are environment configuration;
the default namespace must match the supplied Kubernetes manifests. The fixed
Iris feature schema, application name and approved restart target describe this
assignment's demonstration application.

Repeat real-service verification without rebuilding the deployment:

```bash
python scripts/verify_live.py --approve-training --with-llm --output .runtime/live-verification.json
```

`--approve-training` authorizes two deliberately failed training runs followed by
one successful recovery run. Omit it for read-only checks and inference.
`--with-llm` adds a real four-service conversation and requires the configured
OpenRouter key and provider quota. Neither flag replaces the deployed model.
Prediction verification uses the Kubernetes Service proxy without a fixed local
port and uses the normal kubeconfig discovery rules when `KUBECONFIG` is unset.

Stop foreground commands with Ctrl+C. `docker compose stop` retains service data;
`docker stop assignment03-control-plane` stops the demonstration cluster without
deleting it.
