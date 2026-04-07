"""Scenarios and ScenarioSuites sub-clients for the Release (RLEaaS) SDK.

Accessed via ``client.Scenario`` and ``client.ScenarioSuite``.

A scenario is a structured task definition that governs a single episode —
what the agent should do, which tools it can use, how many steps are
allowed, and which verifier scores it. The quality of your scenarios
directly determines the quality of your trained agent.

Complexity tiers::

    T1 — Single-step / Clear rules     (1–3 steps)
    T2 — Multi-step / Moderate         (4–12 steps)
    T3 — Adversarial / Edge case       (8–20 steps)
    T4 — Systemic stress test          (15–50+ steps)

Example::

    # Browse expert library
    library = client.Scenario.library(
        vertical="FinSim",
        tier=[1, 2],
        domain="wire_transfer",
        tags=["aml", "kyc"],
        limit=20,
    )

    # Import from library into your environment
    scenario = client.Scenario.from_library(
        library_id=library[0]["id"],
        env_id=env.id,
    )

    # Create a custom scenario
    scenario = client.Scenario.create(
        env_id=env.id,
        name="Wire Transfer with AML Escalation",
        tier=2,
        task_prompt="Process a $250,000 wire transfer from account #8823...",
        available_tools=["get_account_balance", "run_aml_check", "initiate_wire_transfer"],
        max_steps=15,
        success_criteria_ref=composite_verifier.id,
        tags=["wire_transfer", "aml", "tier2"],
    )

    # Organize into a suite
    training_suite = client.ScenarioSuite.create(
        name="FinSim Training Suite v1",
        env_id=env.id,
        scenario_ids=[s["id"] for s in tier1 + tier2],
        purpose="training",
    )
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client

SuitePurpose = Literal["training", "evaluation"]


class ScenarioClient:
    """Manages scenario definitions.

    Accessed via ``client.Scenario``.
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    # ------------------------------------------------------------------
    # Expert library
    # ------------------------------------------------------------------

    def library(
        self,
        vertical: Optional[str] = None,
        tier: Optional[List[int]] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Browse the expert scenario library (2,000+ pre-validated scenarios).

        Start here before writing custom scenarios. Each scenario in the
        library has been validated by domain experts and covers realistic
        enterprise workflows.

        Parameters
        ----------
        vertical:
            Filter by environment vertical: ``"FinSim"`` | ``"MedSim"`` | etc.
        tier:
            Filter by complexity tier(s), e.g. ``[1, 2]``.
        domain:
            Domain sub-category, e.g. ``"wire_transfer"``.
        tags:
            Additional tags to filter by, e.g. ``["aml", "kyc"]``.
        limit:
            Maximum number of results to return (default 20).

        Returns
        -------
        list[dict]
            Each entry includes: ``id``, ``tier``, ``title``,
            ``avg_completion_steps``, ``tags``, ``vertical``.

        Example::

            library = client.Scenario.library(
                vertical="FinSim",
                tier=[1, 2],
                domain="wire_transfer",
                tags=["aml", "kyc"],
                limit=20,
            )
            for s in library:
                print(f"{s['id']} | T{s['tier']} | {s['title']} | "
                      f"avg_steps={s.get('avg_completion_steps')}")
        """
        params: Dict[str, Any] = {"limit": limit}
        if vertical:
            params["category"] = vertical
        if tier:
            params["tier"] = ",".join(str(t) for t in tier)
        if domain:
            params["domain"] = domain
        if tags:
            params["tags"] = ",".join(tags)

        data = self._client.get("/api/scenarios", params=params)
        items: List[Dict[str, Any]] = (
            data.get("scenarios", []) if isinstance(data, dict) else data
        )
        return items

    def from_library(
        self,
        library_id: str,
        env_id: Optional[int] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Import a library scenario into your environment.

        Parameters
        ----------
        library_id:
            ``id`` from a :meth:`library` result.
        env_id:
            Target environment numeric ID.
        env_name:
            Target environment name (alternative to *env_id*).

        Returns
        -------
        dict
            The imported scenario definition with its assigned ``id``.

        Example::

            scenario = client.Scenario.from_library(
                library_id=library[0]["id"],
                env_id=env.id,
            )
            print(scenario["id"])  # scen_9f4a1d2e
        """
        env = env_name or (str(env_id) if env_id else "")
        return self._client.post(
            "/api/scenarios",
            json={
                "library_id": library_id,
                "environment": env,
                "environment_id": env_id,
                "source": "library",
            },
        )

    # ------------------------------------------------------------------
    # Custom scenarios
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        env_id: Optional[int] = None,
        env_name: Optional[str] = None,
        tier: int = 1,
        task_prompt: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
        max_steps: int = 20,
        context_injection: Optional[Dict[str, Any]] = None,
        success_criteria_ref: Optional[str] = None,
        tags: Optional[List[str]] = None,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a custom scenario.

        Parameters
        ----------
        name:
            Human-readable scenario title.
        env_id:
            Target environment numeric ID.
        env_name:
            Target environment name (alternative to *env_id*).
        tier:
            Complexity tier 1–4 (default 1).
        task_prompt:
            Natural-language instruction the agent receives at episode start.
            Write it in the same language your production agent will see.
        available_tools:
            Explicit list of tool names available to the agent for this
            scenario. Defaults to the environment's full tool set.
        max_steps:
            Maximum number of tool calls before the episode terminates
            (default 20).
        context_injection:
            Additional context the agent sees at episode start, e.g.
            ``{"regulatory_note": "Transactions over $200K require AML screening."}``.
        success_criteria_ref:
            Verifier ID to use for scoring this scenario's episodes.
            Link after creating the verifier with :meth:`update`.
        tags:
            Searchable tags for organization.
        scenario_id:
            Override the auto-generated scenario ID.

        Returns
        -------
        dict
            The created scenario definition with ``id`` assigned.

        Example::

            scenario = client.Scenario.create(
                env_id=env.id,
                name="Wire Transfer with AML Escalation",
                tier=2,
                task_prompt=(
                    "Process a wire transfer of $250,000 from account #8823 to account #4471. "
                    "Verify KYC status first. Run AML check if amount exceeds $200K. "
                    "Apply the correct fee schedule."
                ),
                available_tools=[
                    "get_account_balance",
                    "check_kyc_status",
                    "run_aml_check",
                    "initiate_wire_transfer",
                    "request_dual_approval",
                ],
                max_steps=15,
                context_injection={
                    "regulatory_note": "Transactions over $200K require AML screening per policy AML-007.",
                    "fee_schedule": "standard_v3",
                },
                success_criteria_ref=composite_verifier.id,
                tags=["wire_transfer", "aml", "kyc", "tier2"],
            )
            print(scenario["id"])  # scen_3c8b5a7f
        """
        sid = scenario_id or f"scen_{uuid.uuid4().hex[:8]}"
        env = env_name or (str(env_id) if env_id else "")

        payload: Dict[str, Any] = {
            "id": sid,
            "name": name,
            "tier": tier,
            "max_steps": max_steps,
            "environment": env,
        }
        if env_id:
            payload["environment_id"] = env_id
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

        result = self._client.post("/api/scenarios", json=payload)
        if isinstance(result, dict) and "id" not in result:
            result["id"] = sid
        return result

    def update(
        self,
        scenario_id: str,
        success_criteria_ref: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Update a scenario, typically to link a verifier after creation.

        Parameters
        ----------
        scenario_id:
            Scenario ID to update.
        success_criteria_ref:
            Verifier ID to link as the scoring criterion.

        Example::

            # Link composite verifier to the scenario
            client.Scenario.update(
                scenario_id=scenario["id"],
                success_criteria_ref=composite_verifier.id,
            )
        """
        payload: Dict[str, Any] = {**fields}
        if success_criteria_ref:
            payload["success_criteria_ref"] = success_criteria_ref
        return self._client.post("/api/scenarios", json={"id": scenario_id, **payload})

    def get(self, scenario_id: str) -> Dict[str, Any]:
        """Get a scenario by ID.

        Searches all scenarios and raises ``KeyError`` if not found.
        """
        all_scenarios = self.list()
        for s in all_scenarios:
            if s.get("id") == scenario_id:
                return s
        raise KeyError(f"Scenario not found: {scenario_id!r}")

    def list(
        self,
        env_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all scenarios with optional filters.

        Parameters
        ----------
        env_name:
            Filter by environment/product name.
        category:
            Filter by vertical/category.

        Example::

            scenarios = client.Scenario.list(env_name="FinSim-Production-Training-v1")
        """
        params: Dict[str, Any] = {}
        if env_name:
            params["product"] = env_name
        if category:
            params["category"] = category

        data = self._client.get("/api/scenarios", params=params or None)
        return data.get("scenarios", []) if isinstance(data, dict) else data

    def delete(self, scenario_id: str) -> Dict[str, Any]:
        """Delete a scenario by ID.

        Returns
        -------
        dict
            ``{"status": "deleted", "id": scenario_id}``
        """
        return self._client.delete(f"/api/scenarios/{scenario_id}")


class ScenarioSuiteClient:
    """Manages scenario suites (versioned collections of scenarios).

    Accessed via ``client.ScenarioSuite``.

    Suites separate training scenarios from held-out evaluation scenarios,
    ensuring your evaluation set is never leaked into the training loop.

    Example::

        training_suite = client.ScenarioSuite.create(
            name="FinSim Training Suite v1",
            env_id=env.id,
            scenario_ids=[s["id"] for s in tier1_scenarios + tier2_scenarios],
            purpose="training",
        )

        eval_suite = client.ScenarioSuite.create(
            name="FinSim Evaluation Suite v1",
            env_id=env.id,
            scenario_ids=[s["id"] for s in eval_scenarios],
            purpose="evaluation",
        )
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def create(
        self,
        name: str,
        scenario_ids: List[str],
        env_id: Optional[int] = None,
        env_name: Optional[str] = None,
        purpose: SuitePurpose = "training",
        suite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a scenario suite.

        Parameters
        ----------
        name:
            Human-readable suite label.
        scenario_ids:
            Ordered list of scenario IDs to include.
        env_id:
            Environment numeric ID.
        env_name:
            Environment name (alternative to *env_id*).
        purpose:
            ``"training"`` — used in the training loop.
            ``"evaluation"`` — held-out; never seen during training.
        suite_id:
            Override the auto-generated suite ID.

        Returns
        -------
        dict
            The created suite definition with ``id`` assigned.

        Example::

            training_suite = client.ScenarioSuite.create(
                name="FinSim Training Suite v1",
                env_id=env.id,
                scenario_ids=[s["id"] for s in tier1 + tier2],
                purpose="training",
            )
        """
        sid = suite_id or f"suite_{uuid.uuid4().hex[:8]}"
        env = env_name or (str(env_id) if env_id else "")

        payload: Dict[str, Any] = {
            "id": sid,
            "name": name,
            "scenario_ids": scenario_ids,
            "purpose": purpose,
            "environment": env,
        }
        if env_id:
            payload["environment_id"] = env_id

        result = self._client.post("/api/scenarios", json=payload)
        if isinstance(result, dict) and "id" not in result:
            result["id"] = sid
            result["name"] = name
            result["scenario_ids"] = scenario_ids
            result["purpose"] = purpose
        return result

    def get(self, suite_id: str) -> Dict[str, Any]:
        """Get a scenario suite by ID."""
        suites = self.list()
        for s in suites:
            if s.get("id") == suite_id:
                return s
        raise KeyError(f"ScenarioSuite not found: {suite_id!r}")

    def list(
        self,
        env_name: Optional[str] = None,
        purpose: Optional[SuitePurpose] = None,
    ) -> List[Dict[str, Any]]:
        """List all scenario suites.

        Parameters
        ----------
        env_name:
            Filter by environment name.
        purpose:
            Filter by ``"training"`` or ``"evaluation"``.
        """
        params: Dict[str, Any] = {"type": "suite"}
        if env_name:
            params["product"] = env_name

        data = self._client.get("/api/scenarios", params=params)
        suites: List[Dict[str, Any]] = (
            data.get("scenarios", []) if isinstance(data, dict) else data
        )
        if purpose:
            suites = [s for s in suites if s.get("purpose") == purpose]
        return suites
