"""Pydantic models for training-related API requests and responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

TrainingStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
TrainingAlgorithm = Literal["PPO", "DQN", "A2C", "GRPO"]


class TrainingRequest(BaseModel):
    """Payload for POST /train/{env_name}."""

    environment_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    algorithm: TrainingAlgorithm = "PPO"
    model: Optional[str] = None
    num_episodes: int = 100
    max_steps: int = 1000
    dataset_url: Optional[str] = None
    verifier_config: Optional[Dict[str, Any]] = None
    run_name: Optional[str] = None
    category: Optional[str] = None
    scenario_id: Optional[str] = None
    actions: Optional[List[str]] = None
    verifier_ids: Optional[List[str]] = None
    verifier_id: Optional[str] = None
    verifier_conditions: Optional[List[Dict[str, Any]]] = None


class TrainingResponse(BaseModel):
    """Immediate response from POST /train/{env_name}."""

    job_id: str
    status: TrainingStatus
    environment_name: str
    message: str


class TrainingJob(BaseModel):
    """Full training job record returned by GET /training/{job_id}."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: TrainingStatus
    environment_name: str
    algorithm: Optional[str] = None
    model: Optional[str] = None
    num_episodes: Optional[int] = None
    current_episode: Optional[int] = None
    progress: Optional[float] = None           # 0.0 – 1.0
    avg_reward: Optional[float] = None
    best_reward: Optional[float] = None
    rewards: List[float] = Field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    run_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Agent(BaseModel):
    """An agent available for training."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    base_model: str
    trainable: bool = True
    compatible_categories: List[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Response from GET /api/training/config."""

    model_config = ConfigDict(extra="allow")

    agents: List[Agent] = Field(default_factory=list)
    algorithms: List[str] = Field(default_factory=list)
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)
