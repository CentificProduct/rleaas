"""Profile sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Profile``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class ProfileClient:
    """Manage the current user's profile."""

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def get(self) -> Dict[str, Any]:
        """Get the current user's profile."""
        data = self._client.get("/api/profile")
        return data if isinstance(data, dict) else {}

    def update(
        self,
        *,
        full_name: Optional[str] = None,
        title: Optional[str] = None,
        bio: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update the current user's profile (upsert semantics)."""
        payload: Dict[str, Any] = {}
        if full_name is not None:
            payload["full_name"] = full_name
        if title is not None:
            payload["title"] = title
        if bio is not None:
            payload["bio"] = bio
        if avatar_url is not None:
            payload["avatar_url"] = avatar_url
        return self._client.put("/api/profile", json=payload)
