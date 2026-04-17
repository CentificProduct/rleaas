"""Deployments sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Deployment``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class DeploymentsClient:
    """Read deployment records and URLs for cloud simulations."""

    def __init__(self, _client: "Client") -> None:
        self._client = _client

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
            Optional filter by URL type, e.g. ``"dashboard"`` or ``"api"``.
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
