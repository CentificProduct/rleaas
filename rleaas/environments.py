"""Environments sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Environment``.

Example::

    env = client.Environment.create(
        name="FinSim-Training-v1",
        vertical="FinSim",
        config={"org_size": "large", "complexity_ceiling": 4},
        tags=["banking", "aml"],
    )
    env.wait_until_ready(timeout=120)
    print(env.status)  # 'ready'
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class EnvironmentResource:
    """A live environment handle returned by :class:`EnvironmentsClient`.

    Exposes instance methods that operate on this specific environment
    (update, deprovision, delete, configure_tools, etc.).
    """

    def __init__(self, data: Dict[str, Any], _client: "Client") -> None:
        self._data = data
        self._client = _client
        self.id: Optional[int] = data.get("id") or data.get("environment_id")
        self.name: str = data.get("name", "")
        self.status: str = data.get("status", "ready")
        self.vertical: Optional[str] = data.get("vertical") or data.get("category")
        self.description: Optional[str] = data.get("description")
        self.tags: List[str] = data.get("tags") or []
        self.metadata: Dict[str, Any] = data.get("metadata") or {}
        self.entity_count: Optional[int] = data.get("entity_count")
        self.tool_count: Optional[int] = data.get("tool_count")
        self.scenario_count: Optional[int] = data.get("scenario_count")
        self.created_at: Optional[str] = data.get("created_at")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _refresh(self) -> "EnvironmentResource":
        """Reload environment data from the API."""
        try:
            updated = self._client.Environment.get(self.name)
            self._data = updated._data
            self.status = updated.status
            self.entity_count = updated.entity_count
            self.tool_count = updated.tool_count
            self.scenario_count = updated.scenario_count
        except Exception:
            pass
        return self

    def wait_until_ready(
        self,
        timeout: float = 120,
        poll_interval: float = 3.0,
    ) -> "EnvironmentResource":
        """Block until environment status is ``'ready'`` or *timeout* elapses.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait (default 120).
        poll_interval:
            Seconds between status polls (default 3).

        Raises
        ------
        TimeoutError
            If the environment is not ready within *timeout* seconds.
        RuntimeError
            If the environment enters a failed/error state.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._refresh()
            if self.status == "ready":
                return self
            if self.status in ("failed", "error"):
                raise RuntimeError(
                    f"Environment {self.name!r} entered status {self.status!r}"
                )
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Environment {self.name!r} did not become ready within {timeout}s "
            f"(current status: {self.status!r})"
        )

    def update(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        system: Optional[str] = None,
        category: Optional[str] = None,
    ) -> "EnvironmentResource":
        """Update environment configuration (creates a new version).

        Parameters
        ----------
        config:
            Vertical-specific config dict (org_size, complexity_ceiling, etc.).
        name:
            Rename this environment instance.
        system:
            Update the system/vertical field.
        category:
            Update the category/domain field.
        """
        if system is not None:
            self._client.put(
                f"/api/environments/{self.name}/system",
                json={"system": system},
            )
        if category is not None:
            self._client.put(
                f"/api/environments/{self.name}/category",
                json={"category": category},
            )
        if config or name:
            payload: Dict[str, Any] = {}
            if config:
                payload["config"] = config
            if name:
                payload["name"] = name
            self._client.put(f"/api/environments/{self.name}", json=payload)
        return self._refresh()

    @staticmethod
    def _auto_cluster_name(provider: str, env_name: str) -> str:
        """Mirror the UI's autoClusterName logic from Step2InfraTools."""
        import re
        prefix = "aks" if provider == "Azure" else "eks" if provider == "AWS" else "gke"
        slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", (env_name or "env").lower())).strip("-")
        return f"{prefix}-{slug}-cluster"

    def deploy(
        self,
        infrastructure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Provision the cloud infrastructure cluster for this environment.

        Calls ``POST /api/deployments`` with the provider-specific settings
        flattened into the payload the API expects.

        Parameters
        ----------
        infrastructure:
            Cloud infrastructure config dict (provider, region, azure/aws/gcp
            sub-dicts, k8s sub-dict).  If omitted, uses the ``infrastructure``
            config stored on this resource at creation time.

        Example::

            env.deploy()                    # use config baked in at create()
            env.deploy(infrastructure={     # or supply/override at call time
                "provider": "AWS",
                "region": "us-east-1",
                "aws": {
                    "account_id": "123456789012",
                    "cluster_name": "my-eks-cluster",
                    "instance_type": "m5.xlarge",
                    "node_count": 3,
                    "network_plugin": "amazon-vpc-cni",
                },
                "k8s": {"k8s_version": "1.32", "namespace": "default",
                        "min_replicas": 1, "max_replicas": 5, "autoscaler": "HPA"},
            })
        """
        infra = infrastructure or self._data.get("infrastructure") or {}
        provider = infra.get("provider", "")
        region = infra.get("region", "")
        k8s = infra.get("k8s", {})

        payload: Dict[str, Any] = {
            "env_name": self.name,
            "provider": provider,
            "region": region,
            "k8s_version": k8s.get("k8s_version", ""),
            "namespace": k8s.get("namespace", "default"),
            "min_replicas": k8s.get("min_replicas", 1),
            "max_replicas": k8s.get("max_replicas", 5),
            "scaling": k8s.get("autoscaler", "HPA"),
        }

        if provider == "Azure":
            azure = infra.get("azure", {})
            cluster_name = azure.get("cluster_name", "") or self._auto_cluster_name(provider, self.name)
            payload.update({
                "subscription_id": azure.get("subscription_id", ""),
                "resource_group": azure.get("resource_group", ""),
                "cluster_name": cluster_name,
                "cluster_label": azure.get("cluster_label", ""),
                "node_pool_name": azure.get("node_pool_name", ""),
                "vm_size": azure.get("vm_size", ""),
                "node_count": azure.get("node_count", 2),
                "network_plugin": azure.get("network_plugin", ""),
            })
        elif provider == "AWS":
            aws = infra.get("aws", {})
            cluster_name = aws.get("cluster_name", "") or self._auto_cluster_name(provider, self.name)
            payload.update({
                "account_id": aws.get("account_id", ""),
                "cluster_name": cluster_name,
                "cluster_label": aws.get("cluster_label", ""),
                "node_group_name": aws.get("node_group_name", ""),
                "instance_type": aws.get("instance_type", ""),
                "node_count": aws.get("node_count", 2),
                "network_plugin": aws.get("network_plugin", ""),
            })
        elif provider == "GCP":
            gcp = infra.get("gcp", {})
            cluster_name = gcp.get("cluster_name", "") or self._auto_cluster_name(provider, self.name)
            payload.update({
                "project_id": gcp.get("project_id", ""),
                "cluster_name": cluster_name,
                "cluster_label": gcp.get("cluster_label", ""),
                "node_pool_name": gcp.get("node_pool_name", ""),
                "machine_type": gcp.get("machine_type", ""),
                "node_count": gcp.get("node_count", 2),
                "network_plugin": gcp.get("network_plugin", ""),
            })

        return self._client.post("/api/deployments", json=payload)

    def deprovision(self) -> Dict[str, Any]:
        """Release compute resources.

        Data assets and trajectories are retained per the tenant retention policy.
        """
        return self._client.post(f"/api/environments/{self.name}/deprovision")

    def delete(self, confirm: bool = False) -> Dict[str, Any]:
        """Hard delete — removes all environment data. **Irreversible.**

        Parameters
        ----------
        confirm:
            Must be ``True`` to proceed.
        """
        if not confirm:
            raise ValueError(
                "Pass confirm=True to permanently delete this environment and all its data."
            )
        return self._client.delete(f"/api/custom-environments/{self.name}")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def configure_tools(
        self,
        access_pattern: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
        approval_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure which tools the agent can access in this environment.

        Parameters
        ----------
        access_pattern:
            Shorthand preset: ``"full"`` | ``"read_only"``.
            Mutually exclusive with *tool_names*.
        tool_names:
            Explicit list of tool IDs/names to enable (matches your production
            agent's exact permission set).
        approval_mode:
            How approval-gate tools behave:
            ``"auto_grant"`` | ``"block"`` | ``"real_approval"``.

        Example::

            # Full access
            env.configure_tools(access_pattern="full")

            # Explicit list matching production permissions
            env.configure_tools(
                tool_names=["get_account_balance", "initiate_wire_transfer"],
                approval_mode="auto_grant",
            )
        """
        payload: Dict[str, Any] = {}
        if access_pattern:
            payload["access_pattern"] = access_pattern
        if tool_names is not None:
            payload["tool_names"] = tool_names
        if approval_mode:
            payload["approval_mode"] = approval_mode
        return self._client.put(
            f"/api/custom-environments/{self.name}", json=payload
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def attach_data(
        self,
        data_asset_id: str,
        injection_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach a data asset to this environment.

        Parameters
        ----------
        data_asset_id:
            ID returned by ``client.DataAsset.generate()`` / ``.upload()``.
        injection_config:
            Optional dict specifying how the data is injected, e.g.
            ``{"entity_type": "transaction"}``.
        """
        return self._client.post(
            f"/api/environments/{self.name}/data",
            json={
                "data_asset_id": data_asset_id,
                "injection_config": injection_config or {},
            },
        )

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------

    def save_repo_config(
        self,
        provider: str = "github",
        repo_url: str = "",
        branch: str = "main",
        context_path: str = "",
        auth_type: str = "none",
        username: str = "",
        token: str = "",
    ) -> Dict[str, Any]:
        """Persist repository configuration for this environment.

        Calls ``PUT /api/environments/{name}/repo``.  The token is stored
        securely server-side and never returned in subsequent reads.

        Parameters
        ----------
        provider:
            Git host: ``"github"`` | ``"gitlab"`` | ``"azure_devops"`` |
            ``"bitbucket"`` | ``"huggingface"`` | ``"other"``.
        repo_url:
            Full HTTPS or SSH clone URL of the repository.
        branch:
            Branch, tag, or commit ref to deploy from (default ``"main"``).
        context_path:
            Optional subdirectory containing deployment manifests.
        auth_type:
            Authentication method: ``"none"`` | ``"pat"`` | ``"basic"``.
        username:
            Git username (required for ``"basic"`` auth only).
        token:
            Personal Access Token or password (required for ``"pat"`` /
            ``"basic"`` auth).

        Returns
        -------
        dict
            Server acknowledgement, e.g. ``{"saved": True}``.

        Example::

            env.save_repo_config(
                provider="github",
                repo_url="https://github.com/org/finsim-env.git",
                branch="main",
                auth_type="pat",
                token="ghp_...",
            )
        """
        payload: Dict[str, Any] = {
            "provider":    provider,
            "repoUrl":     repo_url,
            "branch":      branch,
            "contextPath": context_path,
            "authType":    auth_type,
            "username":    username,
            "token":       token,
        }
        return self._client.put(
            f"/api/environments/{self.name}/repo", json=payload
        )

    def load_repo_config(self) -> Dict[str, Any]:
        """Fetch the saved repository configuration for this environment.

        Calls ``GET /api/environments/{name}/repo``.  The token field is
        redacted in the response.

        Returns
        -------
        dict
            The stored ``RepoConfig`` (``provider``, ``repoUrl``, ``branch``,
            ``contextPath``, ``authType``, ``username``; token is masked).

        Example::

            cfg = env.load_repo_config()
            print(cfg["repoUrl"], cfg["branch"])
        """
        return self._client.get(f"/api/environments/{self.name}/repo")

    def create_scenario(
        self,
        name: str,
        task_prompt: Optional[str] = None,
        tier: int = 1,
        available_tools: Optional[List[str]] = None,
        max_steps: int = 20,
        context_injection: Optional[Dict[str, Any]] = None,
        success_criteria_ref: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a scenario attached to this environment.

        Example::

            scenario = env.create_scenario(
                name="Wire Transfer with AML Escalation",
                tier=2,
                task_prompt="Process a $250K wire transfer...",
                available_tools=["get_account_balance", "run_aml_check"],
                max_steps=15,
            )
        """
        payload: Dict[str, Any] = {
            "name": name,
            "tier": tier,
            "max_steps": max_steps,
        }
        if task_prompt:
            payload["task_prompt"] = task_prompt
        if available_tools is not None:
            payload["available_tools"] = available_tools
        if context_injection:
            payload["context_injection"] = context_injection
        if success_criteria_ref:
            payload["success_criteria_ref"] = success_criteria_ref
        if tags:
            payload["tags"] = tags
        return self._client.post(
            f"/api/environments/{self.name}/scenarios", json=payload
        )

    def attach_verifier(
        self,
        verifier_id: str,
        weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Attach an existing verifier to this environment."""
        return self._client.post(
            f"/api/environments/{self.name}/verifiers",
            json={"verifier_id": verifier_id, "weight": weight},
        )

    def clone_scenarios(self, source_env_name: str) -> Dict[str, Any]:
        """Clone scenarios from another environment into this one."""
        return self._client.post(
            f"/api/environments/{self.name}/clone-scenarios",
            json={"source": source_env_name},
        )

    def clone_verifiers(self, source_env_name: str) -> Dict[str, Any]:
        """Clone verifiers from another environment into this one."""
        return self._client.post(
            f"/api/environments/{self.name}/clone-verifiers",
            json={"source": source_env_name},
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_risk_report(self) -> Dict[str, Any]:
        """Retrieve the compliance risk report for this environment."""
        return self._client.get(f"/environments/{self.name}/risk-report")

    def analyze(self) -> Dict[str, Any]:
        """Run environment analysis (README, file tree, endpoints, models)."""
        return self._client.get(f"/api/environment/{self.name}/analyze")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<EnvironmentResource name={self.name!r} "
            f"id={self.id!r} status={self.status!r}>"
        )


class EnvironmentsClient:
    """Manages environment lifecycle.

    Accessed via ``client.Environment``.

    Example::

        envs = client.Environment.list(vertical="FinSim")
        env  = client.Environment.get("env_8a3f2b1c")
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def create(
        self,
        name: str,
        description: str,
        tools: List[Dict[str, Any]],
        scenarios: List[Dict[str, Any]],
        verifiers: List[str],
        simulations: List[str],
        vertical: Optional[str] = None,
        domain: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        owner: str = "",
        sdk: str = "gradio",
        compute: str = "cpu-basic",
        license: str = "",
        tags: Optional[List[str]] = None,
        infrastructure: Optional[Dict[str, Any]] = None,
        verifier_configs: Optional[Dict[str, Any]] = None,
        source: str = "custom",
        repo: Optional[Dict[str, Any]] = None,
    ) -> "EnvironmentResource":
        """Provision a new synthetic enterprise environment.

        Parameters
        ----------
        name:
            Unique identifier (alphanumeric, hyphens, underscores).
        description:
            Short human-readable description (max 200 chars). Required.
        tools:
            List of tool definition dicts. Supports both SDK-style fields
            (``method``, ``resource``) and UI-style fields (``type``,
            ``parameters`` JSON schema). All fields are forwarded as-is.
        scenarios:
            List of scenario dicts. Supports both SDK-style fields (``tier``,
            ``max_steps``) and UI-style fields (``success``, ``fail``,
            ``product``). All fields are forwarded as-is.
        verifiers:
            List of verifier IDs to attach to this environment. Required.
        simulations:
            List of simulation identifiers for this environment. Required.
        vertical:
            Industry vertical: ``"FinSim"`` | ``"MedSim"`` | ``"DevSim"`` |
            ``"ShipSim"`` | ``"PubSim"`` | ``"ShopSim"``.
        domain:
            UI dropdown value — exactly the four options shown in Step 1:

            * ``"finance"``         → 💰 fin-sim
            * ``"healthcare"``      → 🏥 med-sim
            * ``"enterprise"``      → 📋 dev-sim (Jira / project workflows)
            * ``"human_resources"`` → 👥 hr-sim  (Workday / SAP / ADP)

            When *domain* is given without *vertical*, the SDK automatically
            fills *vertical* using :data:`~rleaas.models.environment.DOMAIN_TO_VERTICAL`
            (e.g. ``"finance"`` → ``"FinSim"``).  You can also supply both
            explicitly — both values are forwarded to the API.
        config:
            Vertical-specific config: ``org_size``, ``complexity_ceiling``,
            ``regulatory_jurisdiction``, ``departments``, ``data_version``.
        owner:
            Owner identifier string.
        sdk:
            Runtime SDK: ``"gradio"`` | ``"docker"`` | ``"static"`` | ``"custom"``.
        compute:
            Compute tier: ``"cpu-basic"`` | ``"cpu-upgrade"`` | ``"gpu-t4"`` | ``"gpu-a10"``.
        license:
            License identifier (e.g. ``"apache-2.0"``).
        tags:
            Searchable tags for organizing environments.
        infrastructure:
            Cloud infrastructure config for cluster provisioning.  Mirrors the
            UI's infrastructure selection (Step 2 – Infrastructure).  Structured
            as a nested dict with a ``provider`` key (``"Azure"`` | ``"AWS"`` |
            ``"GCP"`` | ``"On-Premise"``), a ``region`` key, a provider-specific
            sub-dict (``azure`` / ``aws`` / ``gcp``) with cluster/node settings,
            a ``k8s`` sub-dict with Kubernetes settings, and optional flat fields
            ``cpu``, ``gpu``, ``memory``, ``scaling`` matching the UI infra panel.
            Call :meth:`~EnvironmentResource.deploy` afterwards to actually
            provision the cluster.
        verifier_configs:
            Per-verifier configuration parameters, keyed by verifier ID.
            Mirrors the UI's Step 3 config panel.  Example::

                verifier_configs={
                    "verif_rule_aml_compliance": {"threshold": "0.9", "strict_mode": "true"},
                    "verif_rule_kyc_check": {"threshold": "0.85"},
                }
        source:
            Environment source label (default ``"custom"``). The UI always
            sends ``"custom"`` for user-created environments.

        Example::

            env = client.Environment.create(
                name="FinSim-Production-Training-v1",
                description="Production FinSim environment for AML training",
                tools=[
                    {
                        "id": "ck_get_balance", "name": "get_account_balance",
                        "method": "GET", "resource": "Account",
                        "type": "function", "description": "Fetch account balance.",
                        "parameters": {"type": "object", "properties": {
                            "account_id": {"type": "string"}
                        }},
                        "source": "custom",
                    },
                ],
                scenarios=[{
                    "name": "Wire Transfer AML", "tier": 2, "max_steps": 15,
                    "success": "Transfer completed with passing AML score.",
                    "fail": "Transfer attempted without AML check.",
                    "product": "FinSim",
                }],
                verifiers=["verif_rule_aml_check"],
                verifier_configs={"verif_rule_aml_check": {"threshold": "0.9"}},
                simulations=["finsim-wire-transfer-v1"],
                vertical="FinSim",
                domain="finance",
                config={
                    "org_size": "large",
                    "customer_accounts": 50000,
                    "regulatory_jurisdiction": "US",
                    "complexity_ceiling": 4,
                },
                owner="centific-team",
                license="proprietary",
                tags=["banking", "aml", "production"],
            )
            env.wait_until_ready(timeout=120)
            # Persist repository configuration after creation
            env.save_repo_config(
                provider="github",
                repo_url="https://github.com/org/finsim-env.git",
                branch="main",
                auth_type="pat",
                token="ghp_...",
            )
        """
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "tools": tools,
            "scenarios": scenarios,
            "verifiers": verifiers,
            "simulations": simulations,
            "owner": owner,
            "sdk": sdk,
            "compute": compute,
            "license": license,
            "source": source,
        }
        # Cross-populate domain ↔ vertical so both API fields are always set.
        _domain_to_vertical = {
            "finance": "FinSim", "healthcare": "MedSim",
            "enterprise": "DevSim", "human_resources": "HRSim",
        }
        _vertical_to_domain = {v: k for k, v in _domain_to_vertical.items()}
        resolved_vertical = vertical or (domain and _domain_to_vertical.get(domain))
        resolved_domain = domain or (vertical and _vertical_to_domain.get(vertical))
        if resolved_vertical:
            payload["vertical"] = resolved_vertical
            payload["category"] = resolved_vertical
        if resolved_domain:
            payload["domain"] = resolved_domain
        if config:
            payload["config"] = config
        if tags:
            payload["tags"] = tags
        if infrastructure:
            payload["infrastructure"] = infrastructure
            # Also send as flat `infra` matching the UI's Step 4 payload shape so
            # the server can populate whichever field it reads.
            infra = infrastructure
            payload["infra"] = {
                "provider": infra.get("provider", ""),
                "region": infra.get("region", ""),
                "cpu": infra.get("cpu"),
                "gpu": infra.get("gpu"),
                "memory": infra.get("memory"),
                "scaling": infra.get("scaling", ""),
                **{k: v for k, v in infra.get("azure", {}).items()
                   if k in ("subscription_id", "resource_group", "cluster_name",
                            "cluster_label", "node_pool_name", "vm_size",
                            "node_count", "network_plugin")},
                **{k: v for k, v in infra.get("aws", {}).items()
                   if k in ("account_id", "cluster_name", "cluster_label",
                            "node_group_name", "instance_type", "node_count",
                            "network_plugin")},
                **{k: v for k, v in infra.get("gcp", {}).items()
                   if k in ("project_id", "cluster_name", "cluster_label",
                            "node_pool_name", "machine_type", "node_count",
                            "network_plugin")},
            }
        if verifier_configs:
            payload["verifier_configs"] = verifier_configs
        if repo:
            # Embed repo config in the create payload so the server has it
            # immediately.  Also normalise key names to the camelCase format
            # expected by the UI/API (repo_url → repoUrl, etc.).
            payload["repo"] = {
                "provider":    repo.get("provider",     "github"),
                "repoUrl":     repo.get("repo_url",     repo.get("repoUrl", "")),
                "branch":      repo.get("branch",       "main"),
                "contextPath": repo.get("context_path", repo.get("contextPath", "")),
                "authType":    repo.get("auth_type",    repo.get("authType", "none")),
                "username":    repo.get("username",     ""),
                "token":       repo.get("token",        ""),
            }

        data = self._client.post("/api/custom-environments", json=payload)
        # The server may return a partial dict that omits fields we sent.
        # Seed every sent value with setdefault so EnvironmentResource always
        # reflects what was created, even when the server response is sparse.
        if not isinstance(data, dict):
            data = {}
        data.setdefault("name", name)
        data.setdefault("description", description)
        data.setdefault("owner", owner)
        data.setdefault("sdk", sdk)
        data.setdefault("compute", compute)
        data.setdefault("source", source)
        if resolved_vertical:
            data.setdefault("vertical", resolved_vertical)
            data.setdefault("category", resolved_vertical)
        if resolved_domain:
            data.setdefault("domain", resolved_domain)
        if tags:
            data.setdefault("tags", tags)
        if infrastructure:
            data.setdefault("infrastructure", infrastructure)
        env_resource = EnvironmentResource(data, self._client)

        # Persist repo config via its dedicated endpoint when provided — this
        # mirrors the UI's Step2RepoConfig which calls PUT /api/environments/{name}/repo
        # independently after the environment record is created.
        if repo and (repo.get("repo_url") or repo.get("repoUrl")):
            try:
                env_resource.save_repo_config(
                    provider=repo.get("provider", "github"),
                    repo_url=repo.get("repo_url", repo.get("repoUrl", "")),
                    branch=repo.get("branch", "main"),
                    context_path=repo.get("context_path", repo.get("contextPath", "")),
                    auth_type=repo.get("auth_type", repo.get("authType", "none")),
                    username=repo.get("username", ""),
                    token=repo.get("token", ""),
                )
            except Exception:
                pass  # Non-fatal; the payload already carried repo in the create body

        return env_resource

    def register_environment(
        self,
        name: str,
        description: str,
        tools: List[Dict[str, Any]],
        scenarios: List[Dict[str, Any]],
        verifiers: List[str],
        simulations: List[str],
        vertical: Optional[str] = None,
        domain: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        owner: str = "",
        sdk: str = "gradio",
        compute: str = "cpu-basic",
        license: str = "",
        tags: Optional[List[str]] = None,
        infrastructure: Optional[Dict[str, Any]] = None,
        verifier_configs: Optional[Dict[str, Any]] = None,
        source: str = "custom",
        repo: Optional[Dict[str, Any]] = None,
    ) -> EnvironmentResource:
        """Backward-compatible alias for :meth:`create`.

        Some SDK checklists and integrations refer to this flow as
        ``register_environment()``. The implementation is intentionally
        delegated to :meth:`create` to preserve existing behavior.
        """
        return self.create(
            name=name,
            description=description,
            tools=tools,
            scenarios=scenarios,
            verifiers=verifiers,
            simulations=simulations,
            vertical=vertical,
            domain=domain,
            config=config,
            owner=owner,
            sdk=sdk,
            compute=compute,
            license=license,
            tags=tags,
            infrastructure=infrastructure,
            verifier_configs=verifier_configs,
            source=source,
            repo=repo,
        )

    # ------------------------------------------------------------------
    # Repository configuration
    # ------------------------------------------------------------------

    def validate_repo(
        self,
        repo_url: str,
        branch: str = "main",
        auth_type: str = "none",
        username: str = "",
        token: str = "",
    ) -> Dict[str, Any]:
        """Test connectivity to a Git repository without persisting anything.

        The backend performs a clone probe to verify that the URL is reachable
        and the branch exists.  Call this before :meth:`create` to surface
        credential problems early.

        Parameters
        ----------
        repo_url:
            Full HTTPS or SSH clone URL to probe.
        branch:
            Branch or tag name to verify (default ``"main"``).
        auth_type:
            Authentication method: ``"none"`` | ``"pat"`` | ``"basic"``.
        username:
            Git username (``"basic"`` auth only).
        token:
            Personal Access Token or password (``"pat"`` / ``"basic"`` auth).

        Returns
        -------
        dict
            ``{"ok": True}`` on success, or ``{"ok": False, "message": "..."}``
            on failure.

        Example::

            result = client.Environment.validate_repo(
                repo_url="https://github.com/org/finsim-env.git",
                branch="main",
                auth_type="pat",
                token="ghp_...",
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("message"))
        """
        payload: Dict[str, Any] = {
            "repoUrl":  repo_url,
            "branch":   branch,
            "authType": auth_type,
        }
        if username:
            payload["username"] = username
        if token:
            payload["token"] = token
        return self._client.post("/api/repo/validate", json=payload)

    def save_repo_config(
        self,
        env_name: str,
        provider: str = "github",
        repo_url: str = "",
        branch: str = "main",
        context_path: str = "",
        auth_type: str = "none",
        username: str = "",
        token: str = "",
    ) -> Dict[str, Any]:
        """Persist repository configuration for a named environment.

        Calls ``PUT /api/environments/{env_name}/repo``.

        Parameters
        ----------
        env_name:
            Environment name to attach the repo config to.

        All other parameters mirror :meth:`EnvironmentResource.save_repo_config`.

        Example::

            client.Environment.save_repo_config(
                env_name="FinSim-Production-Training-v1",
                provider="github",
                repo_url="https://github.com/org/finsim-env.git",
                branch="main",
                auth_type="pat",
                token="ghp_...",
            )
        """
        payload: Dict[str, Any] = {
            "provider":    provider,
            "repoUrl":     repo_url,
            "branch":      branch,
            "contextPath": context_path,
            "authType":    auth_type,
            "username":    username,
            "token":       token,
        }
        return self._client.put(f"/api/environments/{env_name}/repo", json=payload)

    def load_repo_config(self, env_name: str) -> Dict[str, Any]:
        """Fetch the saved repository configuration for a named environment.

        Calls ``GET /api/environments/{env_name}/repo``.  The token is masked.

        Returns
        -------
        dict
            The stored ``RepoConfig`` (``provider``, ``repoUrl``, ``branch``,
            ``contextPath``, ``authType``, ``username``).

        Example::

            cfg = client.Environment.load_repo_config("FinSim-Production-Training-v1")
            print(cfg["repoUrl"], cfg["branch"])
        """
        return self._client.get(f"/api/environments/{env_name}/repo")

    def get(self, identifier: str) -> EnvironmentResource:
        """Retrieve a single environment by name or numeric ID.

        Parameters
        ----------
        identifier:
            Environment name (e.g. ``"FinSim-Production-Training-v1"``) or
            numeric ID (e.g. ``"42"``).
        """
        data = self._client.get(f"/api/environments/{identifier}")
        return EnvironmentResource(data, self._client)

    def list(
        self,
        status: Optional[str] = None,
        vertical: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> List[EnvironmentResource]:
        """List all environments with optional filters.

        Parameters
        ----------
        status:
            Filter by lifecycle status, e.g. ``"ready"``.
        vertical:
            Filter by vertical/category, e.g. ``"FinSim"``.
        category:
            Alias for *vertical*.
        limit:
            Maximum environments returned in one call.
        offset:
            Number of environments to skip before collecting results.
        cursor:
            Cursor token for cursor-based pagination backends.

        Example::

            envs = client.Environment.list(status="ready", vertical="FinSim")
            for e in envs:
                print(e.id, e.name, e.status, e.created_at)
        """
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        cat = vertical or category
        if cat:
            params["category"] = cat
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if cursor is not None:
            params["cursor"] = cursor

        data = self._client.get("/api/environments", params=params or None)
        items: List[Dict[str, Any]] = (
            data if isinstance(data, list) else data.get("environments", [])
        )
        return [EnvironmentResource(item, self._client) for item in items]