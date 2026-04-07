"""Verifiers sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Verifier``.

Verifiers are the scoring system at the heart of Release's Reinforcement
Learning from Verifiable Rewards (RLVR) methodology. They translate agent
behavior into reward signals that shape the training loop.

Four verifier types are supported:

- **rule_based** — deterministic binary pass/fail on environment state.
- **trajectory_based** — continuous 0-1 score based on tool-call sequence.
- **llm_judge** — rubric-based quality scoring via a language model.
- **composite** — weighted combination of multiple verifiers with per-type
  minimum floors.

Example::

    # Rule-based compliance check
    rule_v = client.Verifier.create(
        env_id=env.id,
        verifier_type="rule_based",
        name="Wire Transfer Compliance Check",
        config={
            "conditions": [
                "'run_aml_check' in trajectory.tool_calls",
                "'check_kyc_status' in trajectory.tool_calls",
            ],
            "condition_logic": "AND",
            "reward_on_pass": 1.0,
            "reward_on_fail": 0.0,
            "compliance_tags": ["AML", "KYC"],
        },
    )

    # Composite production gate
    composite_v = client.Verifier.create(
        env_id=env.id,
        verifier_type="composite",
        name="Wire Transfer Production Gate",
        config={
            "components": [
                {"verifier_id": rule_v.id,  "weight": 0.40, "floor": 0.80},
                {"verifier_id": traj_v.id,  "weight": 0.30, "floor": 0.60},
                {"verifier_id": llm_v.id,   "weight": 0.20, "floor": 0.50},
            ],
            "overall_pass_threshold": 0.75,
        },
    )
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client

VerifierType = Literal["rule_based", "trajectory_based", "llm_judge", "composite"]


class VerifierResource:
    """A verifier definition returned by :class:`VerifierClient`.

    Instance methods operate on this specific verifier
    (update, duplicate, disable, delete, link to scenario).
    """

    def __init__(self, data: Dict[str, Any], _client: "Client") -> None:
        self._data = data
        self._client = _client
        self.id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.type: str = data.get("type", "")
        self.system: str = data.get("system", "Custom")
        self.environment: str = data.get("environment", "")
        self.status: str = data.get("status", "active")
        self.description: str = data.get("description", "")
        self.logic: Dict[str, Any] = data.get("logic") or {}
        self.metadata: Dict[str, Any] = data.get("metadata") or {}
        self.failure_policy: Dict[str, Any] = data.get("failure_policy") or {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(self, **fields: Any) -> "VerifierResource":
        """Update verifier fields in-place.

        Pass any fields from the :class:`VerifierResource` as keyword args.

        Example::

            verifier.update(
                name="Updated Name",
                description="Revised description",
            )
        """
        payload = {**self._data, **fields}
        result = self._client.put(f"/api/verifiers/{self.id}", json=payload)
        updated = result.get("verifier", result)
        self._data = updated
        self.name = updated.get("name", self.name)
        self.status = updated.get("status", self.status)
        return self

    def duplicate(self) -> "VerifierResource":
        """Create a copy of this verifier with a new ID.

        Example::

            copy = verifier.duplicate()
            print(copy.id)
        """
        result = self._client.post(f"/api/verifiers/{self.id}/duplicate")
        return VerifierResource(result.get("verifier", result), self._client)

    def disable(self) -> "VerifierResource":
        """Toggle this verifier between ``active`` and ``disabled``.

        Example::

            verifier.disable()
            print(verifier.status)  # 'disabled'
        """
        result = self._client.patch(f"/api/verifiers/{self.id}/disable")
        self.status = result.get("status", self.status)
        return self

    def delete(self) -> Dict[str, Any]:
        """Permanently delete this verifier.

        Returns
        -------
        dict
            ``{"status": "deleted", "id": ...}``
        """
        return self._client.delete(f"/api/verifiers/{self.id}")

    def test(
        self,
        state: Optional[Dict[str, Any]] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run a dry-run evaluation of this verifier against sample data.

        Parameters
        ----------
        state:
            Simulated environment state snapshot.
        trajectory:
            List of step dicts representing agent actions to score.

        Returns
        -------
        dict
            ``{"score": float, "passed": bool, "breakdown": {...}}``

        Example::

            result = verifier.test(
                trajectory=[{"tool": "run_aml_check"}, {"tool": "initiate_wire_transfer"}],
            )
            print(result["score"], result["passed"])
        """
        return self._client.post(
            f"/api/verifiers/{self.id}/test",
            json={"state": state or {}, "trajectory": trajectory or []},
        )

    def __repr__(self) -> str:
        return (
            f"<VerifierResource id={self.id!r} name={self.name!r} "
            f"type={self.type!r} status={self.status!r}>"
        )


class VerifierClient:
    """Manages verifier definitions.

    Accessed via ``client.Verifier``.

    The four verifier types map directly to the guide:

    +-----------------------+--------------------------------------------+
    | ``"rule_based"``      | Deterministic conditions on env state.     |
    +-----------------------+--------------------------------------------+
    | ``"trajectory_based"``| Tool-call sequence ordering checks.        |
    +-----------------------+--------------------------------------------+
    | ``"llm_judge"``       | Rubric scoring via a language model.        |
    +-----------------------+--------------------------------------------+
    | ``"composite"``       | Weighted combination with per-type floors.  |
    +-----------------------+--------------------------------------------+
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    # ------------------------------------------------------------------
    # Factory helpers per verifier type
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        name: str,
        verifier_type: VerifierType,
        environment: str,
        config: Dict[str, Any],
        description: str,
        system: str,
        verifier_id: Optional[str],
    ) -> Dict[str, Any]:
        vid = verifier_id or f"verif_{verifier_type[:4]}_{uuid.uuid4().hex[:8]}"
        return {
            "id": vid,
            "name": name,
            "type": verifier_type,
            "system": system,
            "environment": environment,
            "description": description,
            "logic": config,
            "source": "custom",
            "status": "active",
            "failure_policy": {"hard_fail": False, "penalty": 0.0, "log_failure": True},
        }

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        verifier_type: VerifierType,
        environment: str = "",
        env_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        system: str = "Custom",
        verifier_id: Optional[str] = None,
    ) -> "VerifierResource":
        """Create a new verifier definition.

        Parameters
        ----------
        name:
            Human-readable verifier name.
        verifier_type:
            ``"rule_based"`` | ``"trajectory_based"`` | ``"llm_judge"`` |
            ``"composite"``.
        environment:
            Environment name this verifier belongs to.
        env_id:
            Numeric environment ID (alternative to *environment*).
        config:
            Type-specific configuration dict (see examples below).
        description:
            Optional description of what this verifier scores.
        system:
            Vertical system label, e.g. ``"FinSim"``.
        verifier_id:
            Override the auto-generated ID.

        Returns
        -------
        VerifierResource
            The created verifier with its assigned ``id``.

        Examples
        --------
        **Rule-based** (binary pass/fail on conditions)::

            rule_v = client.Verifier.create(
                name="Wire Transfer Compliance Check",
                verifier_type="rule_based",
                environment="FinSim-Production-Training-v1",
                config={
                    "conditions": [
                        "'run_aml_check' in trajectory.tool_calls",
                        "'check_kyc_status' in trajectory.tool_calls",
                        "transfer_amount <= 200000 OR "
                        "'request_dual_approval' in trajectory.tool_calls",
                    ],
                    "condition_logic": "AND",
                    "reward_on_pass": 1.0,
                    "reward_on_fail": 0.0,
                    "compliance_tags": ["AML", "KYC", "SOX"],
                    "explanation_required": True,
                },
            )

        **Trajectory-based** (process order checks)::

            traj_v = client.Verifier.create(
                name="AML Process Order Check",
                verifier_type="trajectory_based",
                environment="FinSim-Production-Training-v1",
                config={
                    "required_actions": [
                        "check_kyc_status",
                        "run_aml_check",
                        "initiate_wire_transfer",
                    ],
                    "ordering_constraints": [
                        {"before": "run_aml_check",    "after": "initiate_wire_transfer"},
                        {"before": "check_kyc_status", "after": "initiate_wire_transfer"},
                    ],
                    "forbidden_actions": ["bypass_compliance_check"],
                    "partial_credit_rubric": {
                        "required_actions_met_ordering_violated": 0.5,
                        "partial_required_actions": 0.3,
                    },
                    "max_allowed_steps": 12,
                    "explanation_required": True,
                },
            )

        **LLM Judge** (rubric quality scoring)::

            llm_v = client.Verifier.create(
                name="Customer Communication Quality",
                verifier_type="llm_judge",
                environment="FinSim-Production-Training-v1",
                config={
                    "judge_model": "claude-3-opus",
                    "evaluation_prompt": (
                        "You are a senior banking compliance officer. "
                        "Evaluate the agent's message on: accuracy, tone, "
                        "absence of unauthorized commitments, clarity."
                    ),
                    "scoring_rubric": {
                        "scale": "1-5",
                        "dimensions": ["accuracy", "tone", "compliance", "clarity"],
                    },
                    "score_to_reward_mapping": {
                        "5": 1.0, "4": 0.8, "3": 0.5, "2": 0.2, "1": 0.0,
                    },
                    "explanation_required": True,
                },
            )

        **Composite** (weighted multi-verifier gate)::

            composite_v = client.Verifier.create(
                name="Wire Transfer Production Gate",
                verifier_type="composite",
                environment="FinSim-Production-Training-v1",
                config={
                    "components": [
                        {"verifier_id": rule_v.id, "weight": 0.40, "floor": 0.80},
                        {"verifier_id": traj_v.id, "weight": 0.30, "floor": 0.60},
                        {"verifier_id": llm_v.id,  "weight": 0.20, "floor": 0.50},
                    ],
                    "overall_pass_threshold": 0.75,
                },
            )
        """
        env_name = environment or (str(env_id) if env_id else "")
        payload = self._build_payload(
            name=name,
            verifier_type=verifier_type,
            environment=env_name,
            config=config or {},
            description=description,
            system=system,
            verifier_id=verifier_id,
        )
        result = self._client.post("/api/verifiers", json=payload)
        data = result.get("verifier", result)
        # Preserve the generated ID if server doesn't echo it back
        if "id" not in data:
            data["id"] = payload["id"]
        return VerifierResource(data, self._client)

    def get(self, verifier_id: str) -> "VerifierResource":
        """Get a single verifier by ID.

        Raises :class:`~rleaas.exceptions.VerifierNotFound` if not found.

        Example::

            v = client.Verifier.get("verif_rule_7d3a")
            print(v.name, v.type, v.status)
        """
        data = self._client.get(f"/api/verifiers/{verifier_id}")
        return VerifierResource(data, self._client)

    def list(
        self,
        environment: Optional[str] = None,
        verifier_type: Optional[str] = None,
        system: Optional[str] = None,
    ) -> List["VerifierResource"]:
        """List verifier definitions with optional filters.

        Parameters
        ----------
        environment:
            Filter by environment name.
        verifier_type:
            Filter by type: ``"rule_based"`` | ``"trajectory_based"`` |
            ``"llm_judge"`` | ``"composite"``.
        system:
            Filter by vertical system label.

        Example::

            # All verifiers for an environment
            verifiers = client.Verifier.list(environment="FinSim-Production-Training-v1")

            # Only rule-based
            rules = client.Verifier.list(verifier_type="rule_based")
        """
        params: Dict[str, Any] = {}
        if environment:
            params["environment"] = environment
        if verifier_type:
            params["type"] = verifier_type
        if system:
            params["system"] = system

        data = self._client.get("/api/verifiers", params=params or None)
        items: List[Dict[str, Any]] = (
            data.get("verifiers", []) if isinstance(data, dict) else data
        )
        return [VerifierResource(item, self._client) for item in items]

    def delete(self, verifier_id: str) -> Dict[str, Any]:
        """Delete a verifier by ID.

        Returns
        -------
        dict
            ``{"status": "deleted", "id": verifier_id}``
        """
        return self._client.delete(f"/api/verifiers/{verifier_id}")

    def test(
        self,
        verifier_id: str,
        state: Optional[Dict[str, Any]] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Dry-run a verifier against sample data.

        Returns
        -------
        dict
            ``{"score": float, "passed": bool, "breakdown": {...}}``

        Example::

            result = client.Verifier.test(
                verifier_id=rule_v.id,
                trajectory=[
                    {"tool": "check_kyc_status"},
                    {"tool": "run_aml_check"},
                    {"tool": "initiate_wire_transfer"},
                ],
            )
            print(result["passed"])
        """
        return self._client.post(
            f"/api/verifiers/{verifier_id}/test",
            json={"state": state or {}, "trajectory": trajectory or []},
        )
