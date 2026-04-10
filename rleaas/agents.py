"""Agents sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Agent``.

Example::

    # Register your agent
    agent = client.Agent.register(
        name="FinSim-Banking-Agent-v1",
        model_ref="your-org/banking-agent:latest",
        framework="openai_compatible",
    )
    print(agent.id)   # agent_q1w2e3r4

    # List available agents
    for a in client.Agent.list():
        print(a.id, a.name, a.base_model)

    # Export a trained checkpoint
    artifact = client.Agent.export(
        checkpoint_id=best_ck["id"],
        format="huggingface",
    )
"""

from __future__ import annotations

import json as _json
import uuid
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client

ExportFormat = Literal["huggingface", "pytorch", "onnx", "openai_compatible"]
AgentFramework = Literal["openai_compatible", "anthropic", "custom"]


class AgentResource:
    """A registered agent returned by :class:`AgentsClient`."""

    def __init__(self, data: Dict[str, Any], _client: "Client") -> None:
        self._data = data
        self._client = _client
        self.id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.base_model: str = data.get("base_model", "")
        self.trainable: bool = bool(data.get("trainable", True))
        self.compatible_categories: List[str] = data.get("compatible_categories") or []
        self.framework: Optional[str] = data.get("framework")
        self.model_ref: Optional[str] = data.get("model_ref")

    def export(
        self,
        checkpoint_id: str,
        format: ExportFormat = "huggingface",
        include_metadata: bool = True,
        include_audit_hash: bool = True,
        download_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a trained checkpoint as a deployable artifact.

        Parameters
        ----------
        checkpoint_id:
            Checkpoint ID from ``job.get_best_checkpoint()``.
        format:
            Export format: ``"huggingface"`` (default) | ``"pytorch"`` |
            ``"onnx"`` | ``"openai_compatible"``.
        include_metadata:
            Include training metadata (env version, algorithm, hyperparams).
        include_audit_hash:
            Include SHA-256 hash for audit trail integrity.
        download_to:
            Local directory path to download artifact into.

        Returns
        -------
        dict
            Export artifact metadata including download URL.

        Example::

            artifact = agent.export(
                checkpoint_id=best_ck["id"],
                format="huggingface",
                include_audit_hash=True,
            )
            # artifact.download_to("./exports/banking-agent-v1/")
        """
        result = self._client.post(
            f"/api/agents/{self.id}/export",
            json={
                "checkpoint_id": checkpoint_id,
                "format": format,
                "include_metadata": include_metadata,
                "include_audit_hash": include_audit_hash,
            },
        )
        if download_to and isinstance(result, dict):
            result["_download_to"] = download_to
        return result

    def __repr__(self) -> str:
        return (
            f"<AgentResource id={self.id!r} name={self.name!r} "
            f"base_model={self.base_model!r} trainable={self.trainable}>"
        )


class AgentsClient:
    """Manages AI agent registration and export.

    Accessed via ``client.Agent``.

    Example::

        agent = client.Agent.register(
            name="FinSim-Banking-Agent-v1",
            model_ref="your-org/banking-agent:latest",
            framework="openai_compatible",
        )
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def register(
        self,
        name: str,
        base_model: str = "",
        model_ref: Optional[str] = None,
        framework: AgentFramework = "openai_compatible",
        trainable: bool = True,
        compatible_categories: Optional[List[str]] = None,
        tool_permissions: Optional[List[str]] = None,
    ) -> AgentResource:
        """Register a new AI agent.

        Parameters
        ----------
        name:
            Human-readable label for this agent version.
        base_model:
            Base model identifier (e.g. ``"Qwen/Qwen3-0.6B"``).
        model_ref:
            Container image or HuggingFace repo ref pointing to your
            agent's implementation (e.g. ``"your-org/banking-agent:latest"``).
            Never transferred to Centific — referenced only for orchestration.
        framework:
            Agent communication protocol: ``"openai_compatible"`` |
            ``"anthropic"`` | ``"custom"``.
        trainable:
            Whether this agent can be fine-tuned (default ``True``).
        compatible_categories:
            Environment verticals this agent is suited for.
        tool_permissions:
            Tool IDs pre-authorized for this agent (can be overridden
            per-environment via ``env.configure_tools()``).

        Returns
        -------
        AgentResource
            Registered agent with ``id`` assigned.

        Example::

            agent = client.Agent.register(
                name="FinSim-Banking-Agent-v1",
                model_ref="your-org/banking-agent:latest",
                framework="openai_compatible",
            )
            print(agent.id)
        """
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        form: Dict[str, Any] = {
            "id": agent_id,
            "name": name,
            "base_model": base_model or model_ref or "",
            "trainable": "true" if trainable else "false",
            "compatible_categories": _json.dumps(compatible_categories or []),
        }

        data = self._client.post("/api/agents", data=form)
        # Server may echo back the registered object or just a status dict
        if not isinstance(data, dict) or "id" not in data:
            payload = {
                "id": agent_id,
                "name": name,
                "base_model": base_model or model_ref or "",
                "trainable": trainable,
                "compatible_categories": compatible_categories or [],
                "model_ref": model_ref,
                "framework": framework,
                "tool_permissions": tool_permissions,
            }
            data = {**payload, **(data or {})}
        return AgentResource(data, self._client)

    def get(self, agent_id: str) -> AgentResource:
        """Get an agent by ID.

        Searches the full agent list and raises ``KeyError`` if not found.
        """
        for agent in self.list():
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Agent not found: {agent_id!r}")

    def list(self) -> List[AgentResource]:
        """List all registered agents (built-in + user-defined).

        Example::

            for a in client.Agent.list():
                print(a.id, a.name, a.base_model, a.trainable)
        """
        data = self._client.get("/api/agents")
        items: List[Dict[str, Any]] = data if isinstance(data, list) else []
        return [AgentResource(item, self._client) for item in items]

    def delete(self, agent_id: str) -> Dict[str, Any]:
        """Delete a registered agent by ID.

        Parameters
        ----------
        agent_id:
            The agent ID to delete (e.g. ``"agent_q1w2e3r4"``).

        Returns
        -------
        dict
            ``{"deleted": True, "agent_id": "..."}``

        Example::

            client.Agent.delete("agent_q1w2e3r4")
        """
        return self._client.delete(f"/api/agents/{agent_id}")

    def export(
        self,
        checkpoint_id: str,
        agent_id: Optional[str] = None,
        format: ExportFormat = "huggingface",
        include_metadata: bool = True,
        include_audit_hash: bool = True,
        download_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a trained agent checkpoint as a deployable artifact.

        This is a convenience method — equivalent to calling
        ``agent.export(checkpoint_id=...)`` on an :class:`AgentResource`.

        Parameters
        ----------
        checkpoint_id:
            Checkpoint ID from ``job.get_best_checkpoint()``.
        agent_id:
            Agent ID to associate the export with (optional).
        format:
            Export format: ``"huggingface"`` | ``"pytorch"`` |
            ``"onnx"`` | ``"openai_compatible"``.
        include_metadata:
            Include training metadata in the artifact.
        include_audit_hash:
            Include SHA-256 hash for compliance audit trail.
        download_to:
            Local directory to download the artifact into.

        Returns
        -------
        dict
            Export artifact metadata including download URL.

        Example::

            artifact = client.Agent.export(
                checkpoint_id=best_ck["id"],
                format="huggingface",
                include_audit_hash=True,
            )
            artifact.download_to("./exports/banking-agent-v1/")
        """
        path = f"/api/agents/{agent_id}/export" if agent_id else "/api/agents/export"
        result = self._client.post(
            path,
            json={
                "checkpoint_id": checkpoint_id,
                "format": format,
                "include_metadata": include_metadata,
                "include_audit_hash": include_audit_hash,
            },
        )
        if download_to and isinstance(result, dict):
            result["_download_to"] = download_to
        return result
