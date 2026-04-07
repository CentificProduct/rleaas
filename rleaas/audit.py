"""AuditLog sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.AuditLog``.

The audit log provides an immutable record of all agent actions, verifier
outcomes, and platform events for governance, compliance, and debugging.
Governance configuration (data retention, PII handling, alert thresholds)
is also managed through this client.

Example::

    # Query recent audit events
    events = client.AuditLog.query(
        environment_name="FinSim-Production-Training-v1",
        event_types=["tool_call", "verifier_result"],
        limit=100,
    )
    for event in events:
        print(event["timestamp"], event["type"], event.get("score"))

    # Configure governance policies
    gov = client.AuditLog.configure_governance(
        environment_name="FinSim-Production-Training-v1",
        data_retention_days=90,
        pii_fields=["account_number", "ssn"],
        alert_thresholds={
            "avg_reward_drop": 0.15,
            "pass_rate_min": 0.80,
        },
    )

    # Retrieve current governance config
    config = client.AuditLog.get_governance()
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class AuditLogClient:
    """Queries audit logs and manages governance configuration.

    Accessed via ``client.AuditLog``.

    The audit log is append-only and immutable — every tool call, verifier
    result, and system event is recorded with a timestamp and SHA-256 chain
    hash for tamper evidence.

    Example::

        events = client.AuditLog.query(
            environment_name="FinSim-Production-Training-v1",
            event_types=["compliance_check"],
            limit=50,
        )
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    # ------------------------------------------------------------------
    # Audit log queries
    # ------------------------------------------------------------------

    def query(
        self,
        environment_name: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        job_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the audit log with filters.

        Parameters
        ----------
        environment_name:
            Filter to events from a specific environment.
        event_types:
            Filter by event type(s), e.g. ``["tool_call", "verifier_result",
            "compliance_check", "human_eval"]``.
        start_time:
            ISO-8601 start timestamp for time-range filter.
        end_time:
            ISO-8601 end timestamp for time-range filter.
        limit:
            Maximum events to return (default 100).
        offset:
            Pagination offset (default 0).
        job_id:
            Filter events related to a specific training job.
        agent_id:
            Filter events for a specific agent.

        Returns
        -------
        list[dict]
            Each event dict includes: ``id``, ``timestamp``, ``type``,
            ``environment``, ``agent_id``, ``details``, ``chain_hash``.

        Example::

            events = client.AuditLog.query(
                environment_name="FinSim-Production-Training-v1",
                event_types=["verifier_result"],
                start_time="2025-01-01T00:00:00Z",
                limit=200,
            )
            failed = [e for e in events if not e.get("passed")]
            print(f"{len(failed)} failed verifier checks")
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if environment_name:
            params["environment_name"] = environment_name
        if event_types:
            params["event_types"] = ",".join(event_types)
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if job_id:
            params["job_id"] = job_id
        if agent_id:
            params["agent_id"] = agent_id

        data = self._client.get("/api/audit", params=params)
        return data if isinstance(data, list) else data.get("events", [])

    def get_event(self, event_id: str) -> Dict[str, Any]:
        """Retrieve a single audit event by ID.

        Parameters
        ----------
        event_id:
            Audit event ID.

        Returns
        -------
        dict
            Full event record with all fields and chain hash.
        """
        return self._client.get(f"/api/audit/{event_id}")

    def export(
        self,
        path: str,
        environment_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """Export audit log to a local file.

        Parameters
        ----------
        path:
            Local file path to write the export to.
        environment_name:
            Scope export to a specific environment.
        start_time:
            ISO-8601 start timestamp.
        end_time:
            ISO-8601 end timestamp.
        format:
            ``"json"`` (default) or ``"csv"``.

        Returns
        -------
        str
            Absolute path to the written file.

        Example::

            path = client.AuditLog.export(
                path="./compliance_audit_Q1.json",
                environment_name="FinSim-Production-Training-v1",
                start_time="2025-01-01T00:00:00Z",
                end_time="2025-03-31T23:59:59Z",
            )
            print(f"Audit log exported to {path}")
        """
        events = self.query(
            environment_name=environment_name,
            start_time=start_time,
            end_time=end_time,
            limit=0,  # server interprets 0 as unlimited when exporting
        )
        abs_path = os.path.abspath(path)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2, ensure_ascii=False)
        return abs_path

    # ------------------------------------------------------------------
    # Governance configuration
    # ------------------------------------------------------------------

    def configure_governance(
        self,
        environment_name: Optional[str] = None,
        data_retention_days: Optional[int] = None,
        pii_fields: Optional[List[str]] = None,
        alert_thresholds: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[str]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Set or update governance policies for an environment.

        Parameters
        ----------
        environment_name:
            Apply governance config to a specific environment. If omitted,
            applies platform-wide.
        data_retention_days:
            Number of days to retain audit and rollout records.
        pii_fields:
            Field names to automatically redact in audit logs.
        alert_thresholds:
            Dict of metric thresholds that trigger alerts, e.g.
            ``{"avg_reward_drop": 0.15, "pass_rate_min": 0.80}``.
        compliance_tags:
            Compliance framework tags to enforce (e.g. ``["SOX", "GDPR"]``).

        Returns
        -------
        dict
            The applied governance config.

        Example::

            gov = client.AuditLog.configure_governance(
                environment_name="FinSim-Production-Training-v1",
                data_retention_days=90,
                pii_fields=["account_number", "ssn", "dob"],
                alert_thresholds={
                    "avg_reward_drop": 0.15,
                    "pass_rate_min": 0.80,
                    "compliance_fail_rate_max": 0.05,
                },
                compliance_tags=["SOX", "AML", "KYC"],
            )
        """
        payload: Dict[str, Any] = {**extra}
        if environment_name:
            payload["environment_name"] = environment_name
        if data_retention_days is not None:
            payload["data_retention_days"] = data_retention_days
        if pii_fields is not None:
            payload["pii_fields"] = pii_fields
        if alert_thresholds is not None:
            payload["alert_thresholds"] = alert_thresholds
        if compliance_tags is not None:
            payload["compliance_tags"] = compliance_tags
        return self._client.post("/governance/configure", json=payload)

    def get_governance(self) -> Dict[str, Any]:
        """Retrieve the current governance configuration.

        Returns
        -------
        dict
            Current governance policies including ``data_retention_days``,
            ``pii_fields``, ``alert_thresholds``, and ``compliance_tags``.

        Example::

            config = client.AuditLog.get_governance()
            print(config["data_retention_days"], config["compliance_tags"])
        """
        return self._client.get("/governance")

    # ------------------------------------------------------------------
    # Auth / identity
    # ------------------------------------------------------------------

    def whoami(self) -> Dict[str, Any]:
        """Return the authenticated user / service-account details.

        Returns
        -------
        dict
            ``{"user_id": str, "email": str, "org": str, "scopes": [...]}``

        Example::

            me = client.AuditLog.whoami()
            print(me["email"], me["org"])
        """
        return self._client.get("/api/me")

    def sso_status(self) -> Dict[str, Any]:
        """Return SSO / IdP integration status for the organisation.

        Returns
        -------
        dict
            ``{"sso_enabled": bool, "provider": str, "domains": [...]}``
        """
        return self._client.get("/api/sso-status")
