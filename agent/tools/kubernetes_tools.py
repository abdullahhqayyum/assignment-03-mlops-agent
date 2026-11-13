"""Namespace-scoped application diagnostics. The dispatcher separately authorizes all mutations."""

from datetime import UTC, datetime

from agent.config import Settings
from kubernetes import client, config


class KubernetesTools:
    def __init__(self, settings: Settings, core=None, apps=None):
        self.settings = settings
        self._core, self._apps = core, apps

    def clients(self):
        if self._core is None:
            config.load_kube_config(config_file=self.settings.kubeconfig, persist_config=False)
            self._core, self._apps = client.CoreV1Api(), client.AppsV1Api()
        return self._core, self._apps

    def scope(self, namespace: str) -> None:
        if namespace != self.settings.namespace:
            raise PermissionError("Namespace is outside the approved Kubernetes namespace")

    def workload_status(self, namespace: str = "mlops-agent") -> dict:
        self.scope(namespace)
        core, apps = self.clients()
        kwargs = {"label_selector": "app=iris-api", "_request_timeout": self.settings.tool_timeout}
        pods = core.list_namespaced_pod(namespace, **kwargs)
        deployments = apps.list_namespaced_deployment(namespace, **kwargs)
        pod_data = []
        for pod in pods.items:
            containers = []
            for status in pod.status.container_statuses or []:
                state = status.state.to_dict() if status.state else {}
                previous = status.last_state.to_dict() if status.last_state else {}
                containers.append(
                    {
                        "name": status.name,
                        "ready": status.ready,
                        "restarts": status.restart_count,
                        "state": state,
                        "last_state": previous,
                    }
                )
            pod_data.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "conditions": [
                        {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
                        for c in pod.status.conditions or []
                    ],
                    "containers": containers,
                }
            )
        deployment_data = [
            {
                "name": d.metadata.name,
                "desired_replicas": d.spec.replicas,
                "ready_replicas": d.status.ready_replicas or 0,
                "available_replicas": d.status.available_replicas or 0,
                "images": [c.image for c in d.spec.template.spec.containers],
                "conditions": [
                    {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
                    for c in d.status.conditions or []
                ],
            }
            for d in deployments.items
        ]
        return {"namespace": namespace, "pods": pod_data, "deployments": deployment_data}

    def services(self, namespace: str = "mlops-agent") -> dict:
        self.scope(namespace)
        core, _ = self.clients()
        services = core.list_namespaced_service(
            namespace, label_selector="", _request_timeout=self.settings.tool_timeout
        )
        data = [
            {
                "name": s.metadata.name,
                "type": s.spec.type,
                "cluster_ip": s.spec.cluster_ip,
                "selector": s.spec.selector,
                "ports": [{"port": p.port, "target_port": p.target_port} for p in s.spec.ports],
            }
            for s in services.items
            if s.metadata.name == "iris-api"
        ]
        endpoints = core.read_namespaced_endpoints(
            "iris-api", namespace, _request_timeout=self.settings.tool_timeout
        )
        ready = [a.ip for subset in endpoints.subsets or [] for a in subset.addresses or []]
        unready = [a.ip for subset in endpoints.subsets or [] for a in subset.not_ready_addresses or []]
        return {
            "namespace": namespace,
            "services": data,
            "ready_endpoints": ready,
            "unready_endpoints": unready,
        }

    def pod_logs(self, namespace: str, pod: str, tail_lines: int = 80) -> dict:
        self.scope(namespace)
        core, _ = self.clients()
        obj = core.read_namespaced_pod(pod, namespace, _request_timeout=self.settings.tool_timeout)
        if (obj.metadata.labels or {}).get("app") != "iris-api":
            raise PermissionError("Pod does not belong to the approved inference application")
        logs = core.read_namespaced_pod_log(
            pod,
            namespace,
            tail_lines=tail_lines,
            limit_bytes=12000,
            _request_timeout=self.settings.tool_timeout,
        )
        return {"pod": pod, "namespace": namespace, "logs": logs, "tail_lines": tail_lines}

    def events(self, namespace: str = "mlops-agent") -> dict:
        self.scope(namespace)
        core, _ = self.clients()
        result = core.list_namespaced_event(namespace, _request_timeout=self.settings.tool_timeout)
        selected = [e for e in result.items if e.involved_object.name.startswith("iris-api")]
        return {
            "namespace": namespace,
            "events": [
                {
                    "object": e.involved_object.name,
                    "type": e.type,
                    "reason": e.reason,
                    "message": e.message,
                    "count": e.count,
                    "last_timestamp": str(e.last_timestamp),
                }
                for e in selected
            ][-40:],
        }

    def app_version(self, namespace: str = "mlops-agent") -> dict:
        import json

        self.scope(namespace)
        core, _ = self.clients()
        result = core.connect_get_namespaced_service_proxy_with_path(
            "iris-api:80", namespace, "version", _request_timeout=self.settings.tool_timeout
        )
        return {
            "namespace": namespace,
            "application": json.loads(result) if isinstance(result, str) else result,
            "source": "Kubernetes Service proxy -> running application's /version",
        }

    def restart(self, namespace: str = "mlops-agent", deployment: str = "iris-api") -> dict:
        self.scope(namespace)
        if deployment != "iris-api":
            raise PermissionError("Only the iris-api Deployment can be restarted")
        _, apps = self.clients()
        stamp = datetime.now(UTC).isoformat()
        apps.patch_namespaced_deployment(
            deployment,
            namespace,
            {
                "spec": {
                    "template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}}
                }
            },
            _request_timeout=self.settings.tool_timeout,
        )
        return {
            "namespace": namespace,
            "deployment": deployment,
            "restart_requested_at": stamp,
            "note": "Restart requested; inspect workload status before claiming recovery",
        }
