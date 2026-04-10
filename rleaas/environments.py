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
        config: Optional[Dict[str, Any]] = None,
        owner: str = "",
        sdk: str = "gradio",
        compute: str = "cpu-basic",
        license: str = "",
        tags: Optional[List[str]] = None,
    ) -> EnvironmentResource:
        """Provision a new synthetic enterprise environment.

        Parameters
        ----------
        name:
            Unique identifier (alphanumeric, hyphens, underscores).
        description:
            Short human-readable description (max 200 chars). Required.
        tools:
            List of tool definition dicts (id, name, method, resource, description). Required.
        scenarios:
            List of scenario dicts defining training tasks. Required.
        verifiers:
            List of verifier IDs to attach to this environment. Required.
        simulations:
            List of simulation identifiers for this environment. Required.
        vertical:
            Industry vertical: ``"FinSim"`` | ``"MedSim"`` | ``"DevSim"`` |
            ``"ShipSim"`` | ``"PubSim"`` | ``"ShopSim"``.
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

        Example::

            env = client.Environment.create(
                name="FinSim-Production-Training-v1",
                description="Production FinSim environment for AML training",
                tools=[
                    {"id": "ck_get_balance", "name": "get_account_balance", "method": "GET", "resource": "Account", "description": "Fetch account balance."},
                    {"id": "ck_run_aml",     "name": "run_aml_check",       "method": "POST","resource": "AML",     "description": "Run AML compliance check."},
                ],
                scenarios=[{"name": "Wire Transfer AML", "tier": 2, "max_steps": 15}],
                verifiers=["verif_rule_aml_check"],
                simulations=["finsim-wire-transfer-v1"],
                vertical="FinSim",
                config={
                    "org_size": "large",
                    "customer_accounts": 50000,
                    "regulatory_jurisdiction": "US",
                    "complexity_ceiling": 4,
                },
                tags=["banking", "aml", "production"],
            )
            env.wait_until_ready(timeout=120)
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
        }
        if vertical:
            payload["vertical"] = vertical
            payload["category"] = vertical
        if config:
            payload["config"] = config
        if tags:
            payload["tags"] = tags

        data = self._client.post("/api/custom-environments", json=payload)
        # Server may return just a status dict; seed name so the resource is usable
        if isinstance(data, dict) and "name" not in data:
            data["name"] = name
        return EnvironmentResource(data, self._client)

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

        data = self._client.get("/api/environments", params=params or None)
        items: List[Dict[str, Any]] = (
            data if isinstance(data, list) else data.get("environments", [])
        )
        return [EnvironmentResource(item, self._client) for item in items]