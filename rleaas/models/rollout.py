"""Pydantic models for rollout and evaluation API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TimelineEvent(BaseModel):
    """A single event in a step's timeline (tool call, model thought, system note, etc.)."""

    model_config = ConfigDict(extra="allow")

    timestamp_ms: Optional[int] = None
    event_type: Optional[str] = None       # SYSTEM | MODEL_THOUGHT | TOOL_CALL | TOOL_RESULT
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None


class RolloutStep(BaseModel):
    """One step within a rollout episode."""

    step: int
    action: Optional[Any] = None
    reward: float = 0.0
    state_summary: Optional[Dict[str, Any]] = None
    reward_breakdown: Optional[Dict[str, float]] = None
    timeline_events: List[TimelineEvent] = Field(default_factory=list)


class VerifierResult(BaseModel):
    """Result from a single verifier check."""

    check: str
    passed: bool
    detail: Optional[str] = None
    score: Optional[float] = None


class Rollout(BaseModel):
    """A complete rollout (episode run) for an environment."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    environment_name: str
    episode_number: int = 1
    steps: List[RolloutStep] = Field(default_factory=list)
    initial_state: Optional[Dict[str, Any]] = None
    final_outcome: Optional[Dict[str, Any]] = None
    final_environment_state: Optional[Dict[str, Any]] = None
    total_reward: float = 0.0
    total_steps: int = 0
    status: str = "completed"              # completed | failed | in_progress
    source: str = "simulation"             # simulation | training
    job_id: Optional[str] = None
    timestamp: Optional[str] = None
    policy_name: Optional[str] = None
    checkpoint_label: Optional[str] = None
    scenario_name: Optional[str] = None
    verifier_results: List[VerifierResult] = Field(default_factory=list)


class RolloutComparison(BaseModel):
    """Pre/post-training rollout comparison from GET /api/rollout-comparison/{env}."""

    model_config = ConfigDict(extra="allow")

    environment_name: str
    pre_training: Optional[Rollout] = None
    post_training: Optional[Rollout] = None
    improvement: Optional[float] = None    # post - pre total_reward


class KPI(BaseModel):
    """A single KPI metric for an environment."""

    model_config = ConfigDict(extra="allow")

    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None


class HumanEvalStepScore(BaseModel):
    """Score for a single step submitted by a human evaluator."""

    step: int
    score: float = Field(..., ge=0.0, le=1.0)
    comment: Optional[str] = None


class HumanEvalRequest(BaseModel):
    """Payload for POST /human-eval/{job_id}."""

    scores: List[HumanEvalStepScore] = Field(default_factory=list)
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None