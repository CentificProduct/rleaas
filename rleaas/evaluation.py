"""Evaluation sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.Evaluation``.

Evaluation runs your trained agent through held-out evaluation scenarios,
scoring it with verifiers to produce an objective quality report. Evaluation
results feed directly into go/no-go deployment decisions.

Workflow::

    # Run a full evaluation
    eval_job = client.Evaluation.run(
        agent_checkpoint_id=best_ck["id"],
        scenario_suite_id=eval_suite["id"],
        verifier_ids=[composite_v.id],
        name="FinSim Eval v1 — post-training",
    )
    report = eval_job.wait_until_complete()
    print(report["overall_score"], report["pass_rate"])

    # Compare two checkpoints head-to-head
    comparison = client.Evaluation.compare(
        agent_checkpoints=[ck_v1["id"], ck_v2["id"]],
        scenario_suite_id=eval_suite["id"],
        verifier_ids=[composite_v.id],
    )

    # Retrieve a rollout from training or evaluation
    rollout = client.Evaluation.get_rollout(env_name="FinSim", rollout_id="roll_abc")

    # Export a signed audit report for compliance
    client.Evaluation.export_audit_report(format="pdf", signature=True)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client

ExportFormat = Literal["pdf", "json", "csv"]


class EvaluationJobResource:
    """A running or completed evaluation job returned by :class:`EvaluationClient`.

    Instance methods poll for completion and retrieve the final report.
    """

    def __init__(self, data: Dict[str, Any], _client: "Client") -> None:
        self._data = data
        self._client = _client
        self.id: str = data.get("id", data.get("job_id", ""))
        self.name: str = data.get("name", "")
        self.status: str = data.get("status", "pending")
        self.environment: str = data.get("environment", "")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Refresh status from the server."""
        try:
            data = self._client.get(f"/api/rollouts/{self.environment}/{self.id}")
            if isinstance(data, dict):
                self._data.update(data)
                self.status = data.get("status", self.status)
        except Exception:
            pass

    def wait_until_complete(
        self,
        timeout: float = 3600.0,
        poll_interval: float = 15.0,
    ) -> Dict[str, Any]:
        """Block until the evaluation job finishes and return the report.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait before raising ``TimeoutError`` (default 3600).
        poll_interval:
            Seconds between status polls (default 15).

        Returns
        -------
        dict
            The final evaluation report including ``overall_score``,
            ``pass_rate``, ``verifier_breakdown``, and ``rollout_ids``.

        Example::

            report = eval_job.wait_until_complete(timeout=1800)
            print(f"Score: {report['overall_score']:.2f}  Pass rate: {report['pass_rate']:.0%}")
        """
        deadline = time.time() + timeout
        terminal = {"completed", "failed", "cancelled", "done", "error"}
        while time.time() < deadline:
            self._refresh()
            if self.status.lower() in terminal:
                return self._data
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Evaluation job {self.id!r} did not complete within {timeout}s "
            f"(last status: {self.status!r})"
        )

    def __repr__(self) -> str:
        return (
            f"<EvaluationJobResource id={self.id!r} name={self.name!r} "
            f"status={self.status!r}>"
        )


class EvaluationClient:
    """Runs evaluation episodes and retrieves rollout data.

    Accessed via ``client.Evaluation``.

    Evaluation uses a held-out scenario suite (``purpose="evaluation"``) to
    measure trained-agent quality without data leakage from the training loop.

    Example::

        eval_job = client.Evaluation.run(
            agent_checkpoint_id=best_ck["id"],
            scenario_suite_id=eval_suite["id"],
            verifier_ids=[composite_v.id],
            name="FinSim Eval — post-GRPO",
        )
        report = eval_job.wait_until_complete(timeout=1800)
        print(report["overall_score"], report["pass_rate"])
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    # ------------------------------------------------------------------
    # Evaluation runs
    # ------------------------------------------------------------------

    def run(
        self,
        agent_checkpoint_id: str,
        scenario_suite_id: str,
        verifier_ids: Optional[List[str]] = None,
        name: str = "",
        environment_name: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationJobResource:
        """Start an evaluation run against a held-out scenario suite.

        Parameters
        ----------
        agent_checkpoint_id:
            Checkpoint ID from ``job.get_best_checkpoint()["id"]``.
        scenario_suite_id:
            Suite ID (``purpose="evaluation"``) to run episodes against.
        verifier_ids:
            Verifier IDs to score each episode. Defaults to the environment's
            attached verifiers if omitted.
        name:
            Human-readable label for this evaluation run.
        environment_name:
            Name of the environment the suite belongs to.
        config:
            Optional overrides for episode config (e.g. ``{"max_steps": 20}``).

        Returns
        -------
        EvaluationJobResource
            Handle to the running evaluation; call ``.wait_until_complete()``
            to block until finished.

        Example::

            eval_job = client.Evaluation.run(
                agent_checkpoint_id=best_ck["id"],
                scenario_suite_id=eval_suite["id"],
                verifier_ids=[composite_v.id],
                name="FinSim Post-Training Evaluation",
            )
        """
        payload: Dict[str, Any] = {
            "agent_checkpoint_id": agent_checkpoint_id,
            "scenario_suite_id": scenario_suite_id,
            "purpose": "evaluation",
        }
        if name:
            payload["name"] = name
        if environment_name:
            payload["environment_name"] = environment_name
        if verifier_ids:
            payload["verifier_ids"] = verifier_ids
        if config:
            payload.update(config)

        data = self._client.post("/api/rollouts", json=payload)
        if not isinstance(data, dict):
            data = {}
        data.setdefault("name", name)
        data.setdefault("environment", environment_name)
        return EvaluationJobResource(data, self._client)

    def get_report(self, evaluation_id: str, environment_name: str = "") -> Dict[str, Any]:
        """Retrieve a completed evaluation report by ID.

        Parameters
        ----------
        evaluation_id:
            Evaluation job or rollout ID.
        environment_name:
            Environment name (required for the scoped endpoint).

        Returns
        -------
        dict
            Evaluation report with ``overall_score``, ``pass_rate``,
            ``verifier_breakdown``, ``scenario_results``, etc.

        Example::

            report = client.Evaluation.get_report(
                evaluation_id=eval_job.id,
                environment_name="FinSim-Production-Training-v1",
            )
        """
        if environment_name:
            return self._client.get(f"/api/rollouts/{environment_name}/{evaluation_id}")
        return self._client.get(f"/api/rollouts-all")

    def compare(
        self,
        agent_checkpoints: List[str],
        scenario_suite_id: str,
        verifier_ids: Optional[List[str]] = None,
        environment_name: str = "",
        baseline_id: Optional[str] = None,
        trained_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare two agent checkpoints on the same scenario suite.

        Use either the ``agent_checkpoints`` list (pair of checkpoint IDs) or
        the ``baseline_id`` / ``trained_id`` params for a rollout comparison.

        Parameters
        ----------
        agent_checkpoints:
            Pair of checkpoint IDs ``[baseline_ck_id, trained_ck_id]``.
        scenario_suite_id:
            Suite to run both agents against.
        verifier_ids:
            Verifiers to use for scoring.
        environment_name:
            Environment name for the comparison endpoint.
        baseline_id:
            Explicit baseline rollout ID (alternative to agent_checkpoints).
        trained_id:
            Explicit trained rollout ID.
        job_id:
            Training job ID to use for context.

        Returns
        -------
        dict
            Side-by-side comparison with ``delta``, ``winner``,
            ``scenario_breakdown``, etc.

        Example::

            comparison = client.Evaluation.compare(
                agent_checkpoints=[baseline_ck["id"], trained_ck["id"]],
                scenario_suite_id=eval_suite["id"],
                verifier_ids=[composite_v.id],
                environment_name="FinSim-Production-Training-v1",
            )
            print(comparison["winner"], comparison["delta"])
        """
        if baseline_id and trained_id and environment_name:
            params: Dict[str, Any] = {
                "baseline_id": baseline_id,
                "trained_id": trained_id,
            }
            if job_id:
                params["job_id"] = job_id
            return self._client.get(
                f"/api/rollout-comparison/{environment_name}",
                params=params,
            )

        # Fall back to posting a comparison job
        payload: Dict[str, Any] = {
            "agent_checkpoints": agent_checkpoints,
            "scenario_suite_id": scenario_suite_id,
            "comparison": True,
        }
        if verifier_ids:
            payload["verifier_ids"] = verifier_ids
        if environment_name:
            payload["environment_name"] = environment_name
        return self._client.post("/api/rollouts", json=payload)

    # ------------------------------------------------------------------
    # Rollout retrieval
    # ------------------------------------------------------------------

    def list_rollouts(
        self,
        environment_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List rollouts, optionally scoped to an environment.

        Parameters
        ----------
        environment_name:
            Filter to a specific environment. If omitted, returns all rollouts.

        Returns
        -------
        list[dict]
            Each dict includes: ``id``, ``status``, ``score``,
            ``scenario_id``, ``created_at``, etc.

        Example::

            rollouts = client.Evaluation.list_rollouts(
                environment_name="FinSim-Production-Training-v1"
            )
            for r in rollouts:
                print(r["id"], r["score"])
        """
        if environment_name:
            data = self._client.get(f"/api/rollouts/{environment_name}")
        else:
            data = self._client.get("/api/rollouts-all")
        return data if isinstance(data, list) else data.get("rollouts", [])

    def get_rollout(
        self,
        rollout_id: str,
        environment_name: str,
    ) -> Dict[str, Any]:
        """Get a single rollout by ID.

        Parameters
        ----------
        rollout_id:
            Rollout ID.
        environment_name:
            Environment the rollout belongs to.

        Returns
        -------
        dict
            Full rollout including step-by-step trajectory.

        Example::

            rollout = client.Evaluation.get_rollout(
                rollout_id="roll_9f4a1d2e",
                environment_name="FinSim-Production-Training-v1",
            )
            for step in rollout["steps"]:
                print(step["tool"], step["reward"])
        """
        return self._client.get(f"/api/rollouts/{environment_name}/{rollout_id}")

    def get_rollout_comparison(
        self,
        environment_name: str,
        baseline_id: str,
        trained_id: str,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve a pre-computed rollout comparison.

        Parameters
        ----------
        environment_name:
            Environment name.
        baseline_id:
            Baseline rollout ID.
        trained_id:
            Trained-agent rollout ID.
        job_id:
            Optional training job context.

        Returns
        -------
        dict
            Comparison with ``delta``, ``winner``, ``per_scenario_diff``.

        Example::

            comp = client.Evaluation.get_rollout_comparison(
                environment_name="FinSim-Production-Training-v1",
                baseline_id="roll_base_001",
                trained_id="roll_trained_001",
            )
            print(comp["winner"], comp["delta"])
        """
        params: Dict[str, Any] = {
            "baseline_id": baseline_id,
            "trained_id": trained_id,
        }
        if job_id:
            params["job_id"] = job_id
        return self._client.get(
            f"/api/rollout-comparison/{environment_name}",
            params=params,
        )

    # ------------------------------------------------------------------
    # Audit report export
    # ------------------------------------------------------------------

    def export_audit_report(
        self,
        evaluation_id: str = "",
        format: ExportFormat = "json",
        include_trajectory_summary: bool = True,
        include_verifier_explanations: bool = True,
        signature: bool = False,
    ) -> Dict[str, Any]:
        """Export a signed audit report for compliance and governance.

        Parameters
        ----------
        evaluation_id:
            Evaluation run to export. Leave blank to export all recent evaluations.
        format:
            Export format: ``"json"`` (default) | ``"pdf"`` | ``"csv"``.
        include_trajectory_summary:
            Include a per-episode step summary.
        include_verifier_explanations:
            Include per-step verifier scoring rationale.
        signature:
            Attach a SHA-256 integrity hash to the report.

        Returns
        -------
        dict
            ``{"report_url": str, "format": str, "sha256": str}``

        Example::

            report = client.Evaluation.export_audit_report(
                evaluation_id=eval_job.id,
                format="pdf",
                signature=True,
            )
            print(report["report_url"])
        """
        payload: Dict[str, Any] = {
            "format": format,
            "include_trajectory_summary": include_trajectory_summary,
            "include_verifier_explanations": include_verifier_explanations,
            "signature": signature,
        }
        if evaluation_id:
            payload["evaluation_id"] = evaluation_id
        return self._client.post("/api/rollouts/export", json=payload)
