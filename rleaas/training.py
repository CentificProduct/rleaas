"""Training sub-client for the Release (RLEaaS) SDK.

Accessed via ``client.TrainingJob``.

Example::

    job = client.TrainingJob.run(
        agent_id=agent.id,
        env_id=env.id,
        scenario_suite_id=training_suite.id,
        algorithm="GRPO",
        config={
            "episodes": 10000,
            "max_steps_per_episode": 20,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "checkpoint_interval": 500,
        },
        name="FinSim-GRPO-Training-Run-1",
    )

    print(job.id)      # job_5e2f8c1a
    print(job.status)  # 'running'

    # Poll until done
    job.wait_until_complete()

    # Stream metrics in a loop
    while job.status == "running":
        metrics = job.get_metrics()
        print(f"Episode {metrics['current_episode']} | reward {metrics['avg_reward']:.3f}")
        import time; time.sleep(30)

    # Checkpoints
    best = job.get_best_checkpoint()
    print(best["id"])
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client


class TrainingJobResource:
    """A live handle to a training job returned by :class:`TrainingClient`.

    Instance methods operate on this specific job (poll status, stream
    metrics, manage checkpoints, submit human eval, cancel).
    """

    def __init__(self, data: Dict[str, Any], _client: "Client") -> None:
        self._data = data
        self._client = _client
        self.id: str = data.get("job_id", "")
        self.status: str = data.get("status", "running")
        self.environment_name: str = data.get("environment_name", "")
        self.algorithm: Optional[str] = data.get("algorithm")
        self.model: Optional[str] = data.get("model") or data.get("agent")
        self.run_name: Optional[str] = data.get("run_name") or data.get("name")
        self.progress: int = data.get("progress", 0)
        self.started_at: Optional[str] = data.get("started_at")

    # ------------------------------------------------------------------
    # Status & metrics
    # ------------------------------------------------------------------

    def _refresh(self) -> "TrainingJobResource":
        """Pull the latest job state from the API."""
        try:
            data = self._client.get(f"/training/{self.id}")
            self._data = data
            self.status = data.get("status", self.status)
            self.progress = data.get("progress", self.progress)
        except Exception:
            pass
        return self

    def get_metrics(self) -> Dict[str, Any]:
        """Return the latest training metrics for this job.

        Refreshes from the API and returns a dict with::

            {
                "job_id":          str,
                "status":          str,
                "current_episode": int,
                "num_episodes":    int,
                "progress":        int,        # 0-100
                "avg_reward":      float | None,
                "success_rate":    float | None,
                "current_tier":    int  | None,
                "results":         dict | None,
            }

        Example::

            metrics = job.get_metrics()
            print(f"Episode {metrics['current_episode']} | "
                  f"Success rate: {metrics['success_rate']:.2%} | "
                  f"Avg reward: {metrics['avg_reward']:.3f}")
        """
        self._refresh()
        results = self._data.get("results") or {}
        return {
            "job_id": self.id,
            "status": self.status,
            "current_episode": self._data.get("progress", 0),
            "num_episodes": self._data.get("num_episodes", 0),
            "progress": self.progress,
            "avg_reward": results.get("avg_reward"),
            "success_rate": results.get("success_rate"),
            "current_tier": results.get("current_tier"),
            "results": results,
        }

    def wait_until_complete(
        self,
        timeout: float = 86400,
        poll_interval: float = 30.0,
        on_progress: Optional[Any] = None,
    ) -> "TrainingJobResource":
        """Block until the job reaches a terminal status.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait (default 86400 = 24 h).
        poll_interval:
            Seconds between status polls (default 30).
        on_progress:
            Optional callable ``fn(metrics: dict)`` called after each poll
            while the job is still running.

        Returns
        -------
        TrainingJobResource
            Self, with status updated to ``"completed"`` (or ``"failed"``).

        Raises
        ------
        TimeoutError
            If the job has not finished within *timeout* seconds.

        Example::

            job.wait_until_complete(
                poll_interval=30,
                on_progress=lambda m: print(
                    f"Episode {m['current_episode']} | reward {m['avg_reward']:.3f}"
                ),
            )
        """
        _terminal = {"completed", "failed", "cancelled", "awaiting_human_eval"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            metrics = self.get_metrics()
            if on_progress:
                try:
                    on_progress(metrics)
                except Exception:
                    pass
            if self.status in _terminal:
                return self
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Training job {self.id!r} did not complete within {timeout}s "
            f"(current status: {self.status!r})"
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved checkpoints for this training run.

        Returns a list of checkpoint dicts, each containing at minimum::

            {
                "id":           str,
                "episode":      int,
                "label":        str,
                "success_rate": float | None,
                "avg_reward":   float | None,
                "created_at":   str  | None,
            }

        Example::

            for ck in job.list_checkpoints():
                print(f"Episode {ck['episode']} | "
                      f"Success rate: {ck.get('success_rate', 0):.2%} | "
                      f"Saved: {ck.get('created_at')}")
        """
        self._refresh()
        raw: List[Dict[str, Any]] = (
            self._data.get("checkpoints")
            or self._data.get("model_metadata", {}).get("checkpoints")
            or []
        )

        # If the server does not expose a checkpoints list, derive one from
        # the job's aggregated results so the method always returns something
        # useful for further SDK calls.
        if not raw and self._data.get("model_path"):
            raw = [
                {
                    "id": f"ckpt_{self.id}_final",
                    "episode": self._data.get("num_episodes", 0),
                    "label": self._data.get("model_path", ""),
                    "success_rate": (self._data.get("results") or {}).get("success_rate"),
                    "avg_reward": (self._data.get("results") or {}).get("avg_reward"),
                    "created_at": self._data.get("started_at"),
                }
            ]
        return raw

    def get_best_checkpoint(
        self,
        metric: str = "avg_reward",
    ) -> Dict[str, Any]:
        """Return the checkpoint with the highest value of *metric*.

        Parameters
        ----------
        metric:
            Key to sort by — ``"avg_reward"`` (default) or
            ``"success_rate"``.

        Returns
        -------
        dict
            The best checkpoint dict from :meth:`list_checkpoints`.

        Raises
        ------
        ValueError
            If no checkpoints are available.

        Example::

            best = job.get_best_checkpoint(metric="composite_verifier_score")
            print(best["id"])  # ckpt_episode_8500
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ValueError(
                f"No checkpoints available for job {self.id!r}. "
                "Ensure training has run at least one checkpoint_interval."
            )
        return max(
            checkpoints,
            key=lambda ck: ck.get(metric) or 0.0,
        )

    # ------------------------------------------------------------------
    # Human evaluation
    # ------------------------------------------------------------------

    def submit_human_eval(
        self,
        decision: str,
        comments: Optional[str] = None,
        step_scores: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Submit a human evaluation for this training job.

        Parameters
        ----------
        decision:
            ``"yes"`` (approve) or ``"no"`` (reject).
        comments:
            Free-text reviewer notes.
        step_scores:
            Per-step scores: ``[{"step_index": int, "score": "correct" |
            "flawed" | "critical_error"}, ...]``.

        Returns
        -------
        dict
            ``{"success": True, "evaluation": {...}, "training_completed": bool}``

        Example::

            result = job.submit_human_eval(
                decision="yes",
                comments="Agent correctly followed AML process.",
                step_scores=[{"step_index": 0, "score": "correct"}],
            )
        """
        payload: Dict[str, Any] = {"decision": decision}
        if comments:
            payload["comments"] = comments
        if step_scores:
            payload["step_scores"] = step_scores
        return self._client.post(f"/human-eval/{self.id}", json=payload)

    # ------------------------------------------------------------------
    # Webhook
    # ------------------------------------------------------------------

    def set_webhook(
        self,
        url: str,
        events: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a webhook URL to receive training events.

        Parameters
        ----------
        url:
            HTTPS endpoint that will receive POST notifications.
        events:
            List of event types to subscribe to, e.g.
            ``["metrics_update", "tier_advance", "checkpoint_saved",
            "job_complete", "reward_collapse_alert"]``.
            Defaults to all events.

        Returns
        -------
        dict
            ``{"status": "registered", "job_id": ..., "url": ...}``

        Example::

            job.set_webhook(
                url="https://your-system.com/webhooks/rleaas",
                events=["job_complete", "reward_collapse_alert"],
            )
        """
        payload: Dict[str, Any] = {"url": url}
        if events:
            payload["events"] = events
        return self._client.post(
            f"/api/training/jobs/{self.id}/webhook", json=payload
        )

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self) -> Dict[str, Any]:
        """Cancel a running training job.

        Returns
        -------
        dict
            ``{"status": "cancelled", "job_id": ...}``
        """
        return self._client.post(f"/api/training/jobs/{self.id}/cancel")

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def list_rollouts(self) -> List[Dict[str, Any]]:
        """List rollouts produced during this training run."""
        try:
            return self._client.get(f"/api/training/runs/{self.id}/rollouts")
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<TrainingJobResource id={self.id!r} "
            f"env={self.environment_name!r} status={self.status!r} "
            f"progress={self.progress}%>"
        )


class TrainingClient:
    """Manages RL training jobs.

    Accessed via ``client.TrainingJob``.

    Example::

        job = client.TrainingJob.run(
            agent_id=agent.id,
            env_id=env.id,
            scenario_suite_id=suite.id,
            algorithm="GRPO",
            config={"episodes": 10000, "max_steps_per_episode": 20},
            name="FinSim-GRPO-Training-Run-1",
        )
    """

    def __init__(self, _client: "Client") -> None:
        self._client = _client

    def run(
        self,
        environment_name: Optional[str] = None,
        env_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        scenario_suite_id: Optional[str] = None,
        algorithm: str = "GRPO",
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        verifier_ids: Optional[List[str]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> TrainingJobResource:
        """Launch a training job.

        Parameters
        ----------
        environment_name:
            Name of the environment to train in (e.g.
            ``"JiraIssueResolution-v0"``).  Use *env_id* if you have the
            numeric ID from :meth:`client.Environment.create`.
        env_id:
            Numeric environment ID (resolves to name internally).
        agent_id:
            Registered agent ID from ``client.Agent.register()``.
        scenario_suite_id:
            Scenario suite to train on.
        algorithm:
            RL algorithm: ``"GRPO"`` (default) | ``"PPO"`` | ``"DQN"`` | ``"A2C"``.
        config:
            Training hyperparameters: ``episodes``, ``max_steps_per_episode``,
            ``learning_rate``, ``batch_size``, ``checkpoint_interval``,
            ``curriculum``, ``compute_quota``.
        name:
            Human-readable label for this run.
        verifier_ids:
            Verifier IDs to score episodes during training.
        resume_from_checkpoint:
            Checkpoint ID to resume training from.

        Returns
        -------
        TrainingJobResource
            Live handle with ``id`` and initial ``status="running"``.

        Example::

            job = client.TrainingJob.run(
                agent_id=agent.id,
                env_id=env.id,
                scenario_suite_id=training_suite.id,
                algorithm="GRPO",
                config={
                    "episodes": 10000,
                    "max_steps_per_episode": 20,
                    "learning_rate": 1e-4,
                    "batch_size": 64,
                    "checkpoint_interval": 500,
                    "curriculum": {
                        "enabled": True,
                        "tier_1_mastery_threshold": 0.80,
                    },
                },
                name="FinSim-GRPO-Training-Run-1",
            )
            print(job.id)      # job_5e2f8c1a
            print(job.status)  # 'running'
        """
        cfg = config or {}
        num_episodes = cfg.pop("episodes", cfg.pop("num_episodes", 100))
        max_steps = cfg.pop("max_steps_per_episode", cfg.pop("max_steps", 1000))

        env_name = environment_name or (str(env_id) if env_id else "default")

        payload: Dict[str, Any] = {
            "algorithm": algorithm,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "config": cfg,
        }
        if agent_id:
            payload["model"] = agent_id
        if name:
            payload["run_name"] = name
        if scenario_suite_id:
            payload["scenario_id"] = scenario_suite_id
        if verifier_ids:
            payload["verifier_ids"] = verifier_ids
        if resume_from_checkpoint:
            payload["resume_from_checkpoint"] = resume_from_checkpoint

        data = self._client.post(f"/train/{env_name}", json=payload)
        return TrainingJobResource(data, self._client)

    def get(self, job_id: str) -> TrainingJobResource:
        """Get a training job by ID.

        Example::

            job = client.TrainingJob.get("job_5e2f8c1a")
            print(job.status)
        """
        data = self._client.get(f"/training/{job_id}")
        return TrainingJobResource(data, self._client)

    def list(self) -> List[TrainingJobResource]:
        """List all training jobs.

        Example::

            for job in client.TrainingJob.list():
                print(job.id, job.status, job.environment_name)
        """
        data = self._client.get("/api/training/jobs")
        jobs: List[Dict[str, Any]] = (
            data.get("jobs", []) if isinstance(data, dict) else data
        )
        return [TrainingJobResource(j, self._client) for j in jobs]

    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job by ID.

        Example::

            client.TrainingJob.cancel("job_5e2f8c1a")
        """
        return self._client.post(f"/api/training/jobs/{job_id}/cancel")
