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

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rleaas.client import Client

SUPPORTED_ALGORITHMS = (
    "GRPO",
    "PPO",
    "SAC",
    "DQN",
    "A2C",
    "A3C",
    "TD3",
    "DDPG",
    "SLM",
)


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

    def _normalize_algorithm(self, algorithm: str) -> str:
        algo = algorithm.strip().upper()
        if algo not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm {algorithm!r}. "
                f"Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS)}."
            )
        return algo

    def _validate_positive_int(self, value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return value

    def _validate_reward_fn(self, reward_fn: Any) -> Any:
        if isinstance(reward_fn, str):
            ref = reward_fn.strip()
            if not ref:
                raise ValueError("reward_fn cannot be empty.")
            # Keep scalar string shape for backend pass-through compatibility.
            return ref
        if isinstance(reward_fn, dict):
            has_type_value = (
                str(reward_fn.get("type", "")).strip() in {"path", "inline"}
                and str(reward_fn.get("value", "")).strip()
            )
            if has_type_value:
                return {
                    "type": str(reward_fn["type"]).strip(),
                    "value": str(reward_fn["value"]).strip(),
                }
            has_path = "path" in reward_fn and str(reward_fn.get("path", "")).strip()
            has_inline = "inline" in reward_fn and str(reward_fn.get("inline", "")).strip()
            if has_path and has_inline:
                raise ValueError("reward_fn must define only one of 'path' or 'inline'.")
            if has_path:
                return {"path": str(reward_fn["path"]).strip()}
            if has_inline:
                return {"inline": str(reward_fn["inline"]).strip()}
        raise ValueError(
            "reward_fn must be a non-empty string path/inline expression, "
            "or a dict with exactly one of {'path', 'inline'}."
        )

    def _validate_simulation(self, simulation: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(simulation, dict):
            raise ValueError("simulation must be a dict when provided.")
        validated: Dict[str, Any] = {}
        if "speed" in simulation:
            speed = simulation["speed"]
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0:
                raise ValueError("simulation.speed must be a positive number.")
            validated["speed"] = speed
        if "seed" in simulation:
            seed = simulation["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("simulation.seed must be a non-negative integer.")
            validated["seed"] = seed
        if "episode_settings" in simulation:
            eps = simulation["episode_settings"]
            if not isinstance(eps, dict):
                raise ValueError("simulation.episode_settings must be a dict.")
            eps_validated: Dict[str, Any] = {}
            if "num_episodes" in eps:
                eps_validated["num_episodes"] = self._validate_positive_int(
                    eps["num_episodes"], "simulation.episode_settings.num_episodes"
                )
            if "max_steps" in eps:
                eps_validated["max_steps"] = self._validate_positive_int(
                    eps["max_steps"], "simulation.episode_settings.max_steps"
                )
            validated["episode_settings"] = eps_validated or eps
        unknown_keys = set(simulation.keys()) - {"speed", "seed", "episode_settings"}
        for key in unknown_keys:
            validated[key] = simulation[key]
        return validated

    def configure_training(
        self,
        algorithm: str = "GRPO",
        max_steps: int = 1000,
        reward_fn: Optional[Any] = None,
        simulation: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build and validate a training configuration before submitting jobs.

        Parameters
        ----------
        algorithm:
            RL algorithm name. Supported: ``GRPO``, ``PPO``, ``SAC``, ``DQN``, ``A2C``.
        max_steps:
            Max steps per episode. Must be a positive integer.
        reward_fn:
            Reward function reference. Supports:
            - file path string (example: ``"rewards/finsim_reward.py"``),
            - inline expression/function string,
            - dict with exactly one of ``{"path": ...}`` or ``{"inline": ...}``.
        simulation:
            Simulation runtime properties dict. Supported keys:
            ``speed``, ``seed``, and ``episode_settings``.
        config:
            Additional training config keys to merge (learning_rate, batch_size, etc.).

        Returns
        -------
        dict
            Validated config payload safe to pass to :meth:`run`.
        """
        validated: Dict[str, Any] = dict(config or {})
        validated["algorithm"] = self._normalize_algorithm(algorithm)
        validated["max_steps"] = self._validate_positive_int(max_steps, "max_steps")
        if reward_fn is not None:
            validated["reward_fn"] = self._validate_reward_fn(reward_fn)
        if simulation is not None:
            validated["simulation"] = self._validate_simulation(simulation)
        return validated

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
            RL algorithm: ``"GRPO"`` (default) | ``"PPO"`` | ``"SAC"`` | ``"DQN"`` | ``"A2C"``.
        config:
            Training hyperparameters: ``episodes``, ``max_steps_per_episode``,
            ``learning_rate``, ``batch_size``, ``checkpoint_interval``,
            ``curriculum``, ``compute_quota``, ``simulation``, ``reward_fn``.
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
        cfg = dict(config or {})
        normalized_algo = self._normalize_algorithm(algorithm)
        num_episodes = cfg.pop("episodes", cfg.pop("num_episodes", 100))
        max_steps = cfg.pop("max_steps_per_episode", cfg.pop("max_steps", 1000))
        num_episodes = self._validate_positive_int(num_episodes, "episodes")
        max_steps = self._validate_positive_int(max_steps, "max_steps")

        if "reward_fn" in cfg:
            cfg["reward_fn"] = self._validate_reward_fn(cfg["reward_fn"])
        if "simulation" in cfg:
            cfg["simulation"] = self._validate_simulation(cfg["simulation"])

        env_name = environment_name or (str(env_id) if env_id else "default")

        payload: Dict[str, Any] = {
            "algorithm": normalized_algo,
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


def _cli_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cli_build_client(args: argparse.Namespace) -> "Client":
    # Local import avoids circular import at module load time.
    from rleaas.client import Client

    cfg = _cli_load_json(args.config)
    api_key = args.api_key or os.environ.get("RLEAAS_API_KEY") or cfg.get("api_key")
    base_url = (args.base_url or cfg.get("base_url", "http://localhost:8000")).rstrip("/")
    if not api_key:
        print(
            "ERROR: No API key provided.\n"
            "Pass --api-key, set RLEAAS_API_KEY, or set api_key in config.json."
        )
        sys.exit(1)
    return Client(api_key=api_key, base_url=base_url)


def _cli_resolve_training_entries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = cfg.get("training")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(
            "Invalid training config. Expected: \"training\": [ { ... }, { ... } ]."
        )
    if len(raw) == 0:
        raise ValueError("No training entries found in config.")
    missing = [i for i, item in enumerate(raw) if "training" not in item]
    if missing:
        raise ValueError(
            f"Each training entry must include 'training' identifier. Missing at indexes: {missing}."
        )
    return raw


def _cli_start_one(training: Dict[str, Any], client: "Client", idx: int | None = None) -> None:
    if "environment_name" not in training:
        raise ValueError("Each training entry must include 'environment_name'.")

    merged_cfg = dict(training.get("config", {}))
    merged_cfg["episodes"] = training.get("episodes", 100)
    if training.get("description"):
        merged_cfg["description"] = training.get("description")

    training_cfg = client.TrainingJob.configure_training(
        algorithm=training.get("algorithm", "PPO"),
        max_steps=training.get("max_steps", 200),
        reward_fn=training.get("reward_fn"),
        simulation=training.get("simulation"),
        config=merged_cfg,
    )

    job = client.TrainingJob.run(
        environment_name=training["environment_name"],
        agent_id=training.get("agent_id"),
        scenario_suite_id=training.get("scenario_id"),
        algorithm=training_cfg["algorithm"],
        config=training_cfg,
        name=training.get("name"),
        verifier_ids=training.get("verifier_ids"),
    )

    training_id = training.get("training")
    prefix = f"[training={training_id} idx={idx}] " if idx is not None else f"[training={training_id}] "
    print(f"{prefix}Training started")
    print(f"job_id: {job.id}")
    print(f"status: {job.status}")
    print(f"environment: {job.environment_name}")
    print(f"algorithm: {job.algorithm or training_cfg['algorithm']}")


def _parse_training_ids(value: Any) -> List[str]:
    if value is None:
        return []
    raw = str(value).strip()
    if not raw:
        raise ValueError("--training cannot be empty.")
    ids = [part.strip() for part in raw.split(",")]
    if any(not item for item in ids):
        raise ValueError("Invalid --training value. Use comma-separated ids like --training 3,4,5.")
    return ids


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Training CLI for config.json + SDK")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config JSON (default: config.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override (defaults to env var or config.json)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL override (defaults to config.json value)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start training job(s) from config.json settings")
    p_start.add_argument(
        "--training",
        dest="training_id",
        default=None,
        help="Run one/more entries by `training` identifier (e.g. --training 3 or --training 3,4,5)",
    )
    p_start.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only one training entry by 0-based index when training is a list",
    )
    p_start.add_argument(
        "--all",
        action="store_true",
        help="Run all training entries when training is a list",
    )

    p_status = sub.add_parser("status", help="Show status for a specific training job")
    p_status.add_argument("--job-id", required=True, help="Training job id")

    p_list = sub.add_parser("list", help="List all training jobs")
    p_list.add_argument(
        "--ids-only",
        action="store_true",
        help="Print only job IDs (one per line)",
    )

    p_cancel = sub.add_parser("cancel", help="Cancel a specific training job")
    p_cancel.add_argument("--job-id", required=True, help="Training job id")

    p_wait = sub.add_parser("wait", help="Wait for a specific training job to complete")
    p_wait.add_argument("--job-id", required=True, help="Training job id")
    p_wait.add_argument("--timeout", type=float, default=86400, help="Wait timeout in seconds")
    p_wait.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds",
    )

    p_metrics = sub.add_parser("metrics", help="Show current metrics for a training job")
    p_metrics.add_argument("--job-id", required=True, help="Training job id")

    p_checkpoints = sub.add_parser("checkpoints", help="List checkpoints for a job")
    p_checkpoints.add_argument("--job-id", required=True, help="Training job id")

    p_rollouts = sub.add_parser("rollouts", help="List rollouts for a job")
    p_rollouts.add_argument("--job-id", required=True, help="Training job id")

    args = parser.parse_args()
    if args.command == "start":
        selectors = [
            bool(args.all),
            args.index is not None,
            args.training_id is not None,
        ]
        if sum(selectors) > 1:
            print("ERROR: Use only one selector: --all OR --index OR --training.")
            sys.exit(1)

    client = _cli_build_client(args)
    try:
        try:
            if args.command == "start":
                cfg = _cli_load_json(args.config)
                entries = _cli_resolve_training_entries(cfg)
                if args.training_id is not None:
                    requested_ids = _parse_training_ids(args.training_id)
                    entry_by_training: Dict[str, Dict[str, Any]] = {}
                    for entry in entries:
                        key = str(entry.get("training"))
                        if key in entry_by_training:
                            raise ValueError(
                                f"Duplicate training identifier found in config: training={key!r}. "
                                "Make training identifiers unique."
                            )
                        entry_by_training[key] = entry
                    for training_id in requested_ids:
                        if training_id not in entry_by_training:
                            raise ValueError(f"No training entry found with training={training_id!r}.")
                        _cli_start_one(entry_by_training[training_id], client)
                elif args.all:
                    for i, entry in enumerate(entries):
                        _cli_start_one(entry, client, idx=i)
                elif args.index is not None:
                    if args.index < 0 or args.index >= len(entries):
                        raise ValueError(f"--index out of range. Expected 0 to {len(entries) - 1}.")
                    _cli_start_one(entries[args.index], client, idx=args.index if len(entries) > 1 else None)
                elif len(entries) > 1:
                    raise ValueError(
                        "Config contains multiple training entries. Use --index <n> to run one or --all to run all."
                    )
                else:
                    _cli_start_one(entries[0], client)
            elif args.command == "status":
                job = client.TrainingJob.get(args.job_id)
                print(f"job_id: {job.id}")
                print(f"status: {job.status}")
                print(f"environment: {job.environment_name}")
                print(f"progress: {job.progress}")
                if job.run_name:
                    print(f"run_name: {job.run_name}")
                if job.algorithm:
                    print(f"algorithm: {job.algorithm}")
            elif args.command == "list":
                jobs = client.TrainingJob.list()
                if not jobs:
                    print("No training jobs found.")
                else:
                    if getattr(args, "ids_only", False):
                        for job in jobs:
                            print(job.id)
                    else:
                        for job in jobs:
                            print(
                                f"{job.id} | status={job.status} | env={job.environment_name} "
                                f"| algorithm={job.algorithm or '-'} | progress={job.progress}"
                            )
            elif args.command == "cancel":
                result = client.TrainingJob.cancel(args.job_id)
                print(f"Cancel request sent for {args.job_id}")
                print(result)
            elif args.command == "wait":
                job = client.TrainingJob.get(args.job_id)
                job.wait_until_complete(timeout=args.timeout, poll_interval=args.poll_interval)
                print(f"job_id: {job.id}")
                print(f"final_status: {job.status}")
            elif args.command == "metrics":
                job = client.TrainingJob.get(args.job_id)
                print(json.dumps(job.get_metrics(), indent=2))
            elif args.command == "checkpoints":
                job = client.TrainingJob.get(args.job_id)
                checkpoints = job.list_checkpoints()
                if not checkpoints:
                    print("No checkpoints found.")
                else:
                    print(json.dumps(checkpoints, indent=2))
            elif args.command == "rollouts":
                job = client.TrainingJob.get(args.job_id)
                rollouts = job.list_rollouts()
                if not rollouts:
                    print("No rollouts found.")
                else:
                    print(json.dumps(rollouts, indent=2))
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    cli_main()

# Backward-compatible aliases used by existing test helpers.
_load_json = _cli_load_json
_resolve_training_entries = _cli_resolve_training_entries
_start_one = _cli_start_one


def _cmd_start(args: argparse.Namespace, client: "Client") -> None:
    cfg = _load_json(args.config)
    entries = _resolve_training_entries(cfg)
    count = len(entries)
    training_id = getattr(args, "training_id", None)
    if training_id is not None:
        requested_ids = _parse_training_ids(training_id)
        entry_by_training: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            key = str(entry.get("training"))
            if key in entry_by_training:
                raise ValueError(
                    f"Duplicate training identifier found in config: training={key!r}. "
                    "Make training identifiers unique."
                )
            entry_by_training[key] = entry
        for requested_id in requested_ids:
            if requested_id not in entry_by_training:
                raise ValueError(f"No training entry found with training={requested_id!r}.")
            _start_one(entry_by_training[requested_id], client)
        return
    if getattr(args, "all", False):
        for i, entry in enumerate(entries):
            _start_one(entry, client, idx=i)
        return
    if getattr(args, "index", None) is not None:
        if args.index < 0 or args.index >= count:
            raise ValueError(f"--index out of range. Expected 0 to {count - 1}.")
        _start_one(entries[args.index], client, idx=args.index if count > 1 else None)
        return
    if count > 1:
        raise ValueError(
            "Config contains multiple training entries. Use --index <n> to run one or --all to run all."
        )
    _start_one(entries[0], client)
