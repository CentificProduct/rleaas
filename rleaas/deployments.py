"""Deployments sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Deployment``.

Covers the full lifecycle of a cluster deployment and its running application:
provisioning, app deployment, URL management, log streaming, model config,
availability testing, and Azure cluster management.

Example::

    # Provision a cluster
    dep = client.Deployment.create(env_name="my-env", provider="Azure", ...)
    dep_id = dep["id"]

    # Poll until cluster is running
    import time
    while True:
        dep = client.Deployment.get(dep_id)
        if dep["status"] in ("running", "failed"):
            break
        time.sleep(20)

    # Deploy the app onto the running cluster
    client.Deployment.deploy_app(dep_id)

    # Stream deployment logs
    cursor = 0
    while True:
        chunk = client.Deployment.stream_logs(dep_id, cursor=cursor)
        for line in chunk["lines"]:
            print(line)
        cursor = chunk["cursor"]
        if chunk["done"]:
            break
        time.sleep(5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class DeploymentsClient:
    """Manages cluster deployments and their running applications.

    Accessed via ``client.Deployment``.
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        provider: str,
        cluster_name: str,
        env_name: Optional[str] = None,
        region: Optional[str] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        account_id: Optional[str] = None,
        project_id: Optional[str] = None,
        cluster_label: Optional[str] = None,
        node_pool_name: Optional[str] = None,
        vm_size: Optional[str] = None,
        instance_type: Optional[str] = None,
        machine_type: Optional[str] = None,
        node_count: Optional[int] = None,
        node_group_name: Optional[str] = None,
        network_plugin: Optional[str] = None,
        k8s_version: Optional[str] = None,
        namespace: Optional[str] = None,
        cpu: Optional[int] = None,
        gpu: Optional[int] = None,
        memory: Optional[int] = None,
        scaling: Optional[str] = None,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        k8s_labels: Optional[Dict[str, str]] = None,
        repo_url: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_provider: Optional[str] = None,
        repo_auth_type: Optional[str] = None,
        repo_username: Optional[str] = None,
        repo_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a new cluster deployment.

        The cluster is provisioned asynchronously; poll :meth:`get` until
        ``status`` is ``"running"`` or ``"failed"``.

        Parameters
        ----------
        provider:
            Cloud provider: ``"Azure"`` | ``"AWS"`` | ``"GCP"``.
        cluster_name:
            Name to assign to the Kubernetes cluster.
        env_name:
            Environment this deployment is for.
        region:
            Cloud region, e.g. ``"eastus"`` (Azure), ``"us-east-1"`` (AWS).
        subscription_id:
            Azure subscription ID.
        resource_group:
            Azure resource group.
        account_id:
            AWS account ID.
        project_id:
            GCP project ID.
        cluster_label:
            Kubernetes label applied to the cluster.
        node_pool_name:
            Azure / GCP node pool name.
        vm_size:
            Azure VM size, e.g. ``"Standard_D4s_v3"``.
        instance_type:
            AWS EC2 instance type, e.g. ``"m5.xlarge"``.
        machine_type:
            GCP machine type, e.g. ``"n1-standard-4"``.
        node_count:
            Number of worker nodes.
        node_group_name:
            AWS node group name.
        network_plugin:
            CNI plugin: ``"azure-cni"`` | ``"kubenet"`` | ``"amazon-vpc-cni"``
            | ``"vpc-native"``.
        k8s_version:
            Kubernetes version string, e.g. ``"1.32"``.
        namespace:
            Kubernetes namespace for RL workloads (default ``"rl-environments"``).
        cpu:
            CPU cores allocated per node.
        gpu:
            GPU units allocated per node.
        memory:
            Memory in GB allocated per node.
        scaling:
            Scaling policy label, e.g. ``"auto-scale"``.
        min_replicas:
            Autoscaler minimum replicas.
        max_replicas:
            Autoscaler maximum replicas.
        k8s_labels:
            Additional Kubernetes labels ``{"key": "value"}``.
        repo_url:
            App repository URL — embedded so the backend can clone immediately
            once the cluster is running.
        repo_branch:
            Branch to check out (default ``"main"``).
        repo_provider:
            Source control provider: ``"github"`` | ``"gitlab"`` | ``"azure-devops"``.
        repo_auth_type:
            Auth mechanism: ``"token"`` | ``"ssh"``.
        repo_username:
            Repository username (for HTTPS auth).
        repo_token:
            Personal access token (for HTTPS auth).

        Returns
        -------
        dict
            The created ``ClusterDeployment`` record (``status: "provisioning"``).

        Example::

            dep = client.Deployment.create(
                env_name="FinSim-Production-Training-v1",
                provider="Azure",
                cluster_name="aks-finsim-cluster",
                subscription_id="5bedbe52-...",
                resource_group="QualityServices-dev",
                region="eastus",
                vm_size="Standard_D4s_v3",
                node_count=3,
                network_plugin="azure-cni",
                k8s_version="1.32",
                namespace="rl-environments",
                min_replicas=1,
                max_replicas=5,
            )
            dep_id = dep["id"]
        """
        payload: Dict[str, Any] = {"provider": provider, "cluster_name": cluster_name}
        for key, val in [
            ("env_name", env_name), ("region", region),
            ("subscription_id", subscription_id), ("resource_group", resource_group),
            ("account_id", account_id), ("project_id", project_id),
            ("cluster_label", cluster_label), ("node_pool_name", node_pool_name),
            ("vm_size", vm_size), ("instance_type", instance_type),
            ("machine_type", machine_type), ("node_count", node_count),
            ("node_group_name", node_group_name), ("network_plugin", network_plugin),
            ("k8s_version", k8s_version), ("namespace", namespace),
            ("cpu", cpu), ("gpu", gpu), ("memory", memory), ("scaling", scaling),
            ("min_replicas", min_replicas), ("max_replicas", max_replicas),
            ("k8s_labels", k8s_labels),
            ("repo_url", repo_url), ("repo_branch", repo_branch),
            ("repo_provider", repo_provider), ("repo_auth_type", repo_auth_type),
            ("repo_username", repo_username), ("repo_token", repo_token),
        ]:
            if val is not None:
                payload[key] = val
        return self._client.post("/api/deployments", json=payload)

    def list(self, env_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by environment name."""
        params: Dict[str, Any] = {}
        if env_name:
            params["env_name"] = env_name
        data = self._client.get("/api/deployments", params=params or None)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            rows = data.get("deployments")
            return rows if isinstance(rows, list) else []
        return []

    def get(self, deployment_id: str) -> Dict[str, Any]:
        """Get a deployment record by ID."""
        data = self._client.get(f"/api/deployments/{deployment_id}")
        return data if isinstance(data, dict) else {}

    def delete(self, deployment_id: str) -> Dict[str, Any]:
        """Delete a deployment record from the database.

        This removes the record only. To also destroy the cloud cluster use
        :meth:`delete_azure_cluster`.

        Returns
        -------
        dict
            ``{"status": "deleted", "id": deployment_id}``
        """
        return self._client.delete(f"/api/deployments/{deployment_id}")

    # ------------------------------------------------------------------
    # Status & sync
    # ------------------------------------------------------------------

    def sync(self, deployment_id: str) -> Dict[str, Any]:
        """Sync deployment status from the cloud provider.

        Use when the background provisioning thread died before persisting
        the final status (e.g. after a server restart).

        Calls ``POST /api/deployments/{id}/sync`` which re-runs
        ``az aks show`` (Azure) or the equivalent and updates the record.

        Returns
        -------
        dict
            The updated ``ClusterDeployment`` record.

        Example::

            dep = client.Deployment.sync(dep_id)
            print(dep["status"])  # "running"
        """
        return self._client.post(f"/api/deployments/{deployment_id}/sync", json={})

    # ------------------------------------------------------------------
    # App deployment
    # ------------------------------------------------------------------

    def deploy_app(self, deployment_id: str) -> Dict[str, Any]:
        """Trigger (or re-trigger) app deployment on a running cluster.

        The backend clones the repository that was embedded at :meth:`create`
        time, auto-detects the framework, builds a container image, and
        deploys it to the cluster. Progress is streamed via :meth:`stream_logs`.

        The API call returns immediately; the deployment runs asynchronously.
        Poll :meth:`get` and inspect ``app_status`` for progress.

        ``app_status`` values:
            ``pending`` → ``cloning`` → ``detecting`` → ``building``
            → ``deploying`` → ``running`` | ``failed``

        Returns
        -------
        dict
            The updated ``ClusterDeployment`` record (``app_status: "cloning"``).

        Example::

            client.Deployment.deploy_app(dep_id)
            import time
            while True:
                dep = client.Deployment.get(dep_id)
                print(dep["app_status"], dep.get("app_status_msg", ""))
                if dep["app_status"] in ("running", "failed"):
                    break
                time.sleep(10)
        """
        return self._client.post(
            f"/api/deployments/{deployment_id}/deploy-app", json={}
        )

    def set_app_status(
        self,
        deployment_id: str,
        status: str,
        message: str = "",
    ) -> Dict[str, Any]:
        """Manually override the ``app_status`` field of a deployment.

        Useful for resetting a stuck ``deploying`` state back to ``pending``
        before re-triggering :meth:`deploy_app`.

        Parameters
        ----------
        deployment_id:
            Deployment record ID.
        status:
            New app status string, e.g. ``"pending"`` | ``"failed"``.
        message:
            Optional human-readable message stored as ``app_status_msg``.

        Returns
        -------
        dict
            The updated ``ClusterDeployment`` record.

        Example::

            client.Deployment.set_app_status(dep_id, "pending", "Resetting for re-deploy")
        """
        return self._client.post(
            f"/api/deployments/{deployment_id}/set-app-status",
            json={"status": status, "message": message},
        )

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def stream_logs(
        self,
        deployment_id: str,
        cursor: int = 0,
        wait_seconds: int = 20,
    ) -> Dict[str, Any]:
        """Long-poll incremental app deployment logs for near-live output.

        Returns a chunk of new log lines starting from *cursor*. Increment
        ``cursor`` by ``chunk["cursor"]`` on each call to avoid re-reading
        lines. Stop when ``chunk["done"]`` is ``True``.

        Parameters
        ----------
        deployment_id:
            Deployment record ID.
        cursor:
            Line offset to start reading from (default 0 for the beginning).
        wait_seconds:
            How long the server should hold the connection waiting for new
            lines before returning an empty chunk (1–30, default 20).

        Returns
        -------
        dict
            Keys: ``lines`` (list[str]), ``cursor`` (int), ``done`` (bool),
            ``app_status``, ``app_status_msg``, ``events``, ``suggestions``,
            ``terminal_event``.

        Example::

            cursor = 0
            while True:
                chunk = client.Deployment.stream_logs(dep_id, cursor=cursor)
                for line in chunk["lines"]:
                    print(line)
                cursor = chunk["cursor"]
                if chunk["done"]:
                    break
                time.sleep(5)
        """
        params: Dict[str, Any] = {
            "cursor": max(0, cursor),
            "wait_seconds": max(1, min(30, wait_seconds)),
        }
        return self._client.get(
            f"/api/deployments/{deployment_id}/logs/stream", params=params
        )

    # ------------------------------------------------------------------
    # URL management
    # ------------------------------------------------------------------

    def get_urls(
        self,
        deployment_id: str,
        *,
        url_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return ``deployment_urls`` entries for a deployment.

        Parameters
        ----------
        deployment_id:
            The deployment record ID.
        url_type:
            Optional filter by URL type: ``"api"`` | ``"dashboard"`` |
            ``"ingress"`` | ``"service"`` | ``"model"`` | ``"custom"``.
        """
        dep = self.get(deployment_id)
        urls = dep.get("deployment_urls")
        items: List[Dict[str, Any]] = urls if isinstance(urls, list) else []
        if url_type:
            want = url_type.lower()
            items = [u for u in items if str((u or {}).get("type") or "").lower() == want]
        return items

    def get_primary_url(
        self,
        deployment_id: str,
        *,
        prefer_type: str = "dashboard",
    ) -> Optional[str]:
        """Return the best URL for a deployment (preferred type first)."""
        urls = self.get_urls(deployment_id)
        if not urls:
            return None
        preferred = [
            u for u in urls if str((u or {}).get("type") or "").lower() == prefer_type.lower()
        ]
        chosen = preferred[0] if preferred else urls[0]
        raw = chosen.get("url") if isinstance(chosen, dict) else None
        return str(raw) if raw else None

    def save_urls(
        self,
        deployment_id: str,
        urls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Replace the deployment URL list for a deployment record.

        Each URL entry should contain at minimum ``url`` (str), ``label``
        (str), and ``type`` (``"api"`` | ``"dashboard"`` | ``"ingress"`` |
        ``"service"`` | ``"model"`` | ``"custom"``).

        Parameters
        ----------
        deployment_id:
            Deployment record ID.
        urls:
            Full replacement list of URL dicts.

        Returns
        -------
        dict
            The updated ``ClusterDeployment`` record.

        Example::

            client.Deployment.save_urls(dep_id, [
                {"label": "API", "url": "https://api.example.com", "type": "api"},
                {"label": "Dashboard", "url": "https://dash.example.com", "type": "dashboard"},
            ])
        """
        return self._client.put(
            f"/api/deployments/{deployment_id}/urls", json={"urls": urls}
        )

    def list_public_urls(self, env_name: str) -> Dict[str, Any]:
        """List all persisted public URLs across all deployments for an environment.

        Returns
        -------
        dict
            Keys: ``env_name``, ``count``, ``urls`` (list with ``url``,
            ``label``, ``type``, ``deployment_id``, ``cluster_name``,
            ``env_name``).

        Example::

            result = client.Deployment.list_public_urls("FinSim-Production-Training-v1")
            for u in result["urls"]:
                print(u["type"], u["url"])
        """
        return self._client.get(
            f"/api/environments/{env_name}/public-urls"
        )

    # ------------------------------------------------------------------
    # Availability testing
    # ------------------------------------------------------------------

    def availability_test(
        self,
        deployment_id: str,
        timeout_seconds: Optional[int] = None,
        interval_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a post-deployment availability test against all registered URLs.

        Sends an HTTP probe to each URL in ``deployment_urls`` and records
        reachability. Useful for smoke-testing a freshly deployed cluster.

        Parameters
        ----------
        deployment_id:
            Deployment record ID.
        timeout_seconds:
            Per-URL probe timeout in seconds.
        interval_seconds:
            Seconds between probes when retrying.

        Returns
        -------
        dict
            Keys: ``acknowledged`` (bool), ``deployment_id``, ``message``,
            ``checks`` (list with ``label``, ``url``, ``reachable``,
            ``status_code``, ``error``), ``elapsed_seconds``.

        Example::

            result = client.Deployment.availability_test(dep_id, timeout_seconds=10)
            for check in result["checks"]:
                status = "✓" if check["reachable"] else "✗"
                print(f"{status} {check['label']} — {check['url']}")
        """
        payload: Dict[str, Any] = {}
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        if interval_seconds is not None:
            payload["interval_seconds"] = interval_seconds
        return self._client.post(
            f"/api/deployments/{deployment_id}/availability-test", json=payload
        )

    # ------------------------------------------------------------------
    # Model endpoint
    # ------------------------------------------------------------------

    def probe_model(
        self,
        deployment_id: str,
        model_url: Optional[str] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Probe a model inference endpoint server-side.

        The backend issues a test request to the model endpoint and returns
        the response. Useful for verifying that a deployed model is responsive
        before starting a training run.

        Parameters
        ----------
        deployment_id:
            Deployment record ID.
        model_url:
            Override the endpoint URL to probe (defaults to the deployment's
            stored model URL).
        model:
            Model identifier to pass in the probe request.
        prompt:
            Test prompt to send (defaults to a short built-in ping prompt).
        stream:
            Whether to request a streaming response.

        Returns
        -------
        dict
            Keys: ``ok`` (bool), ``endpoint`` (str), ``status_code`` (int),
            ``result`` (dict).

        Example::

            result = client.Deployment.probe_model(dep_id)
            if result["ok"]:
                print("Model endpoint is live:", result["endpoint"])
        """
        payload: Dict[str, Any] = {}
        if model_url is not None:
            payload["model_url"] = model_url
        if model is not None:
            payload["model"] = model
        if prompt is not None:
            payload["prompt"] = prompt
        if stream:
            payload["stream"] = stream
        return self._client.post(
            f"/api/deployments/{deployment_id}/probe-model", json=payload
        )

    def configure_model(
        self,
        deployment_id: str,
        model_id: str,
        hf_token: str = "",
        namespace: str = "",
        endpoint_url: str = "",
    ) -> Dict[str, Any]:
        """Auto-detect model type and deploy the inference service to the cluster.

        Detects whether *model_id* is a HuggingFace model or an Ollama model,
        deploys the appropriate inference service to the cluster, and persists
        the live endpoint URL to ``deployment_urls``.

        Parameters
        ----------
        deployment_id:
            Deployment record ID.
        model_id:
            Model identifier, e.g. ``"Qwen/Qwen3-0.6B"`` (HuggingFace) or
            ``"llama3"`` (Ollama).
        hf_token:
            HuggingFace access token (required for gated models).
        namespace:
            Kubernetes namespace to deploy into (defaults to deployment's
            namespace).
        endpoint_url:
            Pre-known endpoint URL to register directly (skips deployment if
            set).

        Returns
        -------
        dict
            The updated ``ClusterDeployment`` record with the model endpoint
            URL added to ``deployment_urls``.

        Example::

            client.Deployment.configure_model(
                dep_id,
                model_id="Qwen/Qwen3-0.6B",
                hf_token="hf_...",
            )
        """
        return self._client.post(
            f"/api/deployments/{deployment_id}/configure-model",
            json={
                "model_id": model_id,
                "hf_token": hf_token,
                "namespace": namespace,
                "endpoint_url": endpoint_url,
            },
        )

    # ------------------------------------------------------------------
    # Cloud config
    # ------------------------------------------------------------------

    def get_cloud_config(self) -> Dict[str, Any]:
        """Fetch the platform cloud infrastructure configuration.

        Returns the current cloud config used to populate provider dropdowns
        (regions, VM sizes, K8s versions, autoscaler policies, network plugins)
        in the UI Step 2 — Infrastructure panel.

        Returns
        -------
        dict
            Keys: ``TRAINING_ALGORITHMS``, ``CLOUD_PROVIDERS``, ``REGIONS``,
            ``PROVIDER_REGIONS``, ``PROVIDER_INSTANCE_TYPES``, ``K8S_VERSIONS``,
            ``NETWORK_PLUGINS``, ``AUTOSCALER_POLICIES``.

        Example::

            cfg = client.Deployment.get_cloud_config()
            print(cfg["PROVIDER_REGIONS"]["Azure"])  # ['eastus', 'westus2', ...]
            print(cfg["K8S_VERSIONS"])               # ['1.32', '1.31', ...]
        """
        return self._client.get("/api/cloud-config")

    def update_cloud_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Persist an updated cloud infrastructure configuration.

        Only the fields present in *patch* overwrite the stored values;
        omitted fields are preserved. Changes take effect immediately and
        are reflected in the UI Step 2 — Infrastructure panel.

        Parameters
        ----------
        patch:
            Partial ``CloudConfig`` dict. Any of the top-level keys can be
            updated: ``K8S_VERSIONS`` (list), ``AUTOSCALER_POLICIES`` (list),
            ``PROVIDER_REGIONS`` (dict), ``PROVIDER_INSTANCE_TYPES`` (dict),
            ``NETWORK_PLUGINS`` (dict).

        Returns
        -------
        dict
            The merged ``CloudConfig`` after applying the patch.

        Example::

            client.Deployment.update_cloud_config({
                "K8S_VERSIONS": ["1.32", "1.31", "1.30"],
                "PROVIDER_REGIONS": {
                    "Azure": ["eastus", "westus2", "swedencentral"],
                    "AWS":   ["us-east-1", "eu-west-1"],
                    "GCP":   ["us-central1", "europe-west1"],
                },
            })
        """
        return self._client.put("/api/cloud-config", json=patch)

    # ------------------------------------------------------------------
    # Azure cluster management
    # ------------------------------------------------------------------

    def list_azure_clusters(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List live AKS clusters directly from Azure.

        Queries the Azure API in real time (via ``az aks list``).  Does not
        use the local database — returns current cloud state only.

        Parameters
        ----------
        subscription_id:
            Azure subscription ID to query.
        resource_group:
            Optional resource group filter.

        Returns
        -------
        list[dict]
            Each entry mirrors ``ClusterDeployment`` fields populated from
            Azure's response.

        Example::

            clusters = client.Deployment.list_azure_clusters(
                subscription_id="5bedbe52-...",
                resource_group="QualityServices-dev",
            )
            for c in clusters:
                print(c["cluster_name"], c["status"])
        """
        params: Dict[str, Any] = {"subscription_id": subscription_id}
        if resource_group:
            params["resource_group"] = resource_group
        data = self._client.get("/api/azure/clusters", params=params)
        return data if isinstance(data, list) else []

    def delete_azure_cluster(
        self,
        cluster_name: str,
        subscription_id: str,
        resource_group: str,
    ) -> Dict[str, Any]:
        """Delete an AKS cluster from Azure.

        Runs ``az aks delete --no-wait`` in the background. Azure typically
        takes 3–5 minutes to complete the deletion.

        Parameters
        ----------
        cluster_name:
            Name of the AKS cluster to delete.
        subscription_id:
            Azure subscription ID the cluster belongs to.
        resource_group:
            Azure resource group the cluster belongs to.

        Returns
        -------
        dict
            ``{"deleted": cluster_name, "resource_group": resource_group, "status": "deleting"}``

        Example::

            client.Deployment.delete_azure_cluster(
                cluster_name="aks-test-cluster-4",
                subscription_id="5bedbe52-...",
                resource_group="QualityServices-dev",
            )
        """
        params: Dict[str, Any] = {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
        }
        return self._client.delete(
            f"/api/azure/clusters/{cluster_name}?subscription_id={subscription_id}"
            f"&resource_group={resource_group}"
        )
