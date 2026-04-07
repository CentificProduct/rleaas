"""Pydantic models for verifier-related API requests and responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FailurePolicy(BaseModel):
    hard_fail: bool = False
    penalty: float = 0.0
    log_failure: bool = True


class Verifier(BaseModel):
    """A verifier definition stored in the registry."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    name: str
    type: str                              # rule-based | trajectory-based | llm-judge
    system: str = "Custom"
    environment: str
    env_name: str = Field(default="", alias="envName")
    source: str = "custom"                 # custom | cloned | built-in
    version: int = 1
    status: str = "active"
    used_in_scenarios: List[str] = Field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    logic: Dict[str, Any] = Field(default_factory=dict)
    example_input: Dict[str, Any] = Field(default_factory=dict)
    example_output: Dict[str, Any] = Field(default_factory=dict)
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class VerifierConfigRequest(BaseModel):
    """Legacy verifier configuration payload."""

    verifier_type: str
    environment_name: str
    weights: Dict[str, float]
    thresholds: Optional[Dict[str, float]] = None
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


class VerifierListResponse(BaseModel):
    """Response from GET /api/verifiers."""

    verifiers: List[Verifier] = Field(default_factory=list)
    count: int = 0