"""Tools sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Tools``.

Example::

    # List all tools for an environment
    tools = client.Tools.list(env_id=env.id)

    # Filter by category
    read_tools  = client.Tools.list(env_id=env.id, category="read")
    write_tools = client.Tools.list(env_id=env.id, category="write")

    # Inspect a specific tool
    tool = client.Tools.get("initiate_wire_transfer", env_id=env.id)
    print(tool["description"])
    print(tool["compliance_tags"])

    # Register a custom tool
    custom = client.Tools.register(
        env_id=env.id,
        name="custom_fraud_scorer",
        description="Scores transactions for fraud risk.",
        compliance_tags=["AML"],
    )
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class ToolsClient:
    """Manages tool discovery and registration.

    Accessed via ``client.Tools``.
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def list(
        self,
        env_id: Optional[int] = None,
        env_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List tools, optionally scoped to an environment.

        Parameters
        ----------
        env_id:
            Numeric environment ID to filter by.
        env_name:
            Environment name to filter by.
        category:
            Tool category to filter by (``"read"`` | ``"write"`` |
            ``"approval"`` | ``"search"`` | ``"notify"`` | ``"system"``).

        Example::

            tools      = client.Tools.list(env_id=env.id)
            read_tools = client.Tools.list(env_id=env.id, category="read")
        """
        params: Dict[str, Any] = {}
        if env_id is not None:
            params["environment_id"] = str(env_id)
        if env_name:
            params["environment"] = env_name

        data = self._client.get("/api/tools", params=params or None)
        tools: List[Dict[str, Any]] = (
            data.get("tools", []) if isinstance(data, dict) else data
        )

        if category:
            tools = [
                t for t in tools
                if (t.get("type") or t.get("category") or "").lower() == category.lower()
            ]
        return tools

    def get(
        self,
        tool_id: str,
        env_id: Optional[int] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a single tool by ID or name.

        Parameters
        ----------
        tool_id:
            Tool ID or tool name to look up.
        env_id:
            Narrow the search to a specific environment (numeric ID).
        env_name:
            Narrow the search to a specific environment (name).

        Raises
        ------
        KeyError
            If no matching tool is found.

        Example::

            tool = client.Tools.get("initiate_wire_transfer", env_id=env.id)
            print(tool["description"])
            print(tool.get("compliance_tags"))  # ['AML', 'KYC', 'SOX']
        """
        tools = self.list(env_id=env_id, env_name=env_name)
        for tool in tools:
            if tool.get("id") == tool_id or tool.get("name") == tool_id:
                return tool
        raise KeyError(f"Tool not found: {tool_id!r}")

    def register(
        self,
        env_id: int,
        name: str,
        tool_type: str = "custom",
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[str]] = None,
        authorization_required: bool = False,
        docker_image: Optional[str] = None,
        spec_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a custom MCP tool for an environment.

        Parameters
        ----------
        env_id:
            Target environment numeric ID.
        name:
            Tool name (snake_case recommended, e.g. ``"custom_fraud_scorer"``).
        tool_type:
            Category: ``"custom"`` | ``"read"`` | ``"write"`` | ``"approval"``.
        description:
            Human-readable description of what the tool does.
        parameters:
            JSON Schema describing the tool's input parameters.
        compliance_tags:
            Regulatory tags, e.g. ``["AML", "KYC"]``.
        authorization_required:
            Whether the tool requires dual-approval before execution.
        docker_image:
            Container image serving this tool's implementation.
        spec_path:
            Local path to an OpenAPI 3.0 spec file (loaded and sent as-is).

        Returns
        -------
        dict
            ``{"id": ..., "name": ..., "status": "created"}``

        Example::

            tool = client.Tools.register(
                env_id=env.id,
                name="custom_fraud_scorer",
                description="Scores transactions for fraud risk.",
                compliance_tags=["AML"],
                authorization_required=False,
            )
            print(tool["id"])  # tool_custom_a1b2c3
        """
        tool_id = f"tool_custom_{uuid.uuid4().hex[:8]}"
        tool_payload: Dict[str, Any] = {
            "id": tool_id,
            "name": name,
            "type": tool_type,
            "description": description,
            "parameters": parameters or {},
            "source": "custom",
        }
        if compliance_tags:
            tool_payload["compliance_tags"] = compliance_tags
        if authorization_required:
            tool_payload["authorization_required"] = authorization_required
        if docker_image:
            tool_payload["docker_image"] = docker_image

        if spec_path:
            import json as _json
            import os
            with open(spec_path, "r", encoding="utf-8") as fh:
                tool_payload["spec"] = _json.load(fh) if spec_path.endswith(".json") else fh.read()

        result = self._client.post(
            "/api/tools",
            json={"environment_id": env_id, "tools": [tool_payload]},
        )
        return {"id": tool_id, "name": name, **result}

    def upsert(
        self,
        tools: List[Dict[str, Any]],
        env_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bulk upsert multiple tools at once.

        Parameters
        ----------
        tools:
            List of tool dicts. Each must contain at least ``id`` and ``name``.
        env_id:
            Environment ID applied to all tools (unless a tool has its own
            ``environment_id`` field).

        Returns
        -------
        dict
            ``{"status": "ok", "created": N, "updated": M}``
        """
        payload: Dict[str, Any] = {"tools": tools}
        if env_id is not None:
            payload["environment_id"] = env_id
        return self._client.post("/api/tools", json=payload)

    def update(self, tool_id: str, **fields: Any) -> Dict[str, Any]:
        """Partially update a tool by ID.

        Pass any subset of ``name``, ``type``, ``description``,
        ``parameters``, ``environment``, ``environment_id``, ``source``
        as keyword arguments.

        Returns
        -------
        dict
            ``{"status": "ok", "id": tool_id}``
        """
        return self._client.put(f"/api/tools/{tool_id}", json=fields)

    def delete(self, tool_id: str) -> Dict[str, Any]:
        """Delete a tool by ID.

        Returns
        -------
        dict
            ``{"status": "deleted", "id": tool_id}``
        """
        return self._client.delete(f"/api/tools/{tool_id}")
