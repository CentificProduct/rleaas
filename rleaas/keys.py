"""API Key management sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Keys``.

API keys let SDK clients authenticate without SSO session cookies.  Each key
is a long-lived Bearer token stored as a SHA-256 hash on the server — the
plaintext key is returned **only once** at creation time.

Example::

    # Create a new key (save the returned "key" value — it won't appear again)
    result = client.Keys.create(name="ci-pipeline")
    print(result["key"])   # rleaas_sk_...  ← store in CI secret vault

    # List existing keys (metadata only)
    keys = client.Keys.list()
    for k in keys:
        print(k["id"], k["name"], k["prefix"], k["is_active"])

    # Revoke a key by ID
    client.Keys.revoke(key_id="3")
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class KeysClient:
    """Manage API keys for SDK authentication.

    Accessed via ``client.Keys``.
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def create(self, name: str) -> Dict[str, Any]:
        """Create a new API key.

        Parameters
        ----------
        name:
            Human-readable label for this key (e.g. ``"ci-pipeline"``).

        Returns
        -------
        dict
            Includes ``id``, ``name``, ``key`` (raw Bearer token — shown **once
            only**), ``prefix``, ``is_active``, ``created_at``.

        Example::

            result = client.Keys.create(name="prod-agent")
            import os
            os.environ["RLEAAS_API_KEY"] = result["key"]
        """
        return self._client.post("/api/keys", json={"name": name})

    def list(self) -> List[Dict[str, Any]]:
        """List all API keys (metadata only — raw key never returned).

        Returns
        -------
        list[dict]
            Each entry includes ``id``, ``name``, ``prefix``, ``is_active``,
            ``created_at``, ``last_used_at``.

        Example::

            keys = client.Keys.list()
            active = [k for k in keys if k["is_active"]]
            print(f"{len(active)} active keys")
        """
        return self._client.get("/api/keys")

    def revoke(self, key_id: str) -> Dict[str, Any]:
        """Permanently revoke (deactivate) an API key.

        Parameters
        ----------
        key_id:
            Numeric string ID of the key to revoke (from :meth:`list`).

        Returns
        -------
        dict
            ``{"revoked": True, "id": key_id}``

        Example::

            keys = client.Keys.list()
            old = next(k for k in keys if k["name"] == "old-pipeline")
            client.Keys.revoke(old["id"])
        """
        return self._client.delete(f"/api/keys/{key_id}")
