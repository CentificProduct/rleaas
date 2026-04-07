"""Metrics sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Metrics``.

Provides access to KPI snapshots, historical KPI series, and real-time
metric queries for environment and training job observability.

KPIs are computed from rollout data and reflect agent performance across
all episodes run against an environment's evaluation suite.

Example::

    # Latest KPI snapshot
    kpis = client.Metrics.get_kpis(environment_name="FinSim-Production-Training-v1")
    print(kpis["avg_reward"], kpis["pass_rate"])

    # Historical KPI series (last 50 checkpoints)
    history = client.Metrics.get_kpi_history(
        environment_name="FinSim-Production-Training-v1",
        limit=50,
    )
    for entry in history:
        print(entry["checkpoint"], entry["avg_reward"])

    # Clear stale history records
    client.Metrics.clear_kpi_history()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class MetricsClient:
    """Queries KPIs and training metrics for environments and jobs.

    Accessed via ``client.Metrics``.

    KPIs are computed from rollout data and refreshed automatically after
    each training checkpoint. Use ``refresh=True`` to force a recompute.

    Example::

        kpis = client.Metrics.get_kpis(
            environment_name="FinSim-Production-Training-v1",
            refresh=True,
        )
        print(kpis["avg_reward"], kpis["pass_rate"], kpis["aml_compliance_rate"])
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    # ------------------------------------------------------------------
    # KPI snapshots
    # ------------------------------------------------------------------

    def get_kpis(
        self,
        environment_name: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve the latest KPI snapshot for an environment.

        Parameters
        ----------
        environment_name:
            Filter to a specific environment. Returns platform-wide KPIs
            if omitted.
        refresh:
            Force a recompute from raw rollout data (default ``False``).

        Returns
        -------
        dict
            KPI snapshot including: ``avg_reward``, ``pass_rate``,
            ``avg_completion_steps``, and any domain-specific metrics
            (e.g. ``aml_compliance_rate``, ``kyc_check_rate``).

        Example::

            kpis = client.Metrics.get_kpis(
                environment_name="FinSim-Production-Training-v1",
                refresh=True,
            )
            print(f"Avg reward: {kpis['avg_reward']:.3f}")
            print(f"Pass rate:  {kpis['pass_rate']:.1%}")
        """
        params: Dict[str, Any] = {}
        if environment_name:
            params["environment_name"] = environment_name
        if refresh:
            params["refresh"] = "true"
        return self._client.get("/kpis", params=params or None)

    def get_kpi_history(
        self,
        environment_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical KPI entries (one per checkpoint evaluation).

        Parameters
        ----------
        environment_name:
            Filter by environment name.
        limit:
            Maximum number of history entries (default 50).
        offset:
            Number of entries to skip for pagination (default 0).

        Returns
        -------
        list[dict]
            Each entry includes: ``checkpoint``, ``timestamp``,
            ``avg_reward``, ``pass_rate``, and domain-specific metrics.

        Example::

            history = client.Metrics.get_kpi_history(
                environment_name="FinSim-Production-Training-v1",
                limit=100,
            )
            for entry in history:
                print(entry["checkpoint"], entry["avg_reward"])

            # Plot learning curve
            rewards = [e["avg_reward"] for e in history]
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if environment_name:
            params["environment_name"] = environment_name
        data = self._client.get("/kpis/history", params=params)
        return data if isinstance(data, list) else data.get("history", [])

    def clear_kpi_history(self) -> Dict[str, Any]:
        """Delete all stored KPI history records.

        Use with care — this permanently removes all historical KPI data
        and cannot be undone.

        Returns
        -------
        dict
            ``{"status": "cleared"}``

        Example::

            client.Metrics.clear_kpi_history()
        """
        return self._client.delete("/kpis/history")

    # ------------------------------------------------------------------
    # MCP / Agent tool observability
    # ------------------------------------------------------------------

    def list_agent_tools(self) -> List[Dict[str, Any]]:
        """List available MCP agent tools for observability queries.

        Returns
        -------
        list[dict]
            Each dict describes an available observability tool with
            ``name``, ``description``, and ``parameters``.

        Example::

            tools = client.Metrics.list_agent_tools()
            for t in tools:
                print(t["name"], t["description"])
        """
        data = self._client.get("/api/agent/tools")
        return data if isinstance(data, list) else data.get("tools", [])

    def invoke_agent_tool(
        self,
        tool: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke an MCP observability tool by name.

        Parameters
        ----------
        tool:
            Tool name as returned by :meth:`list_agent_tools`.
        params:
            Tool-specific parameters dict.

        Returns
        -------
        dict
            Tool output — structure depends on the specific tool invoked.

        Example::

            result = client.Metrics.invoke_agent_tool(
                tool="get_environment_kpis",
                params={"environment_name": "FinSim-Production-Training-v1"},
            )
        """
        return self._client.post(
            "/api/agent/invoke",
            json={"tool": tool, "params": params or {}},
        )

    def agent_health(self) -> Dict[str, Any]:
        """Check the health of the MCP observability agent.

        Returns
        -------
        dict
            ``{"status": "ok", ...}``
        """
        return self._client.get("/api/agent/health")
