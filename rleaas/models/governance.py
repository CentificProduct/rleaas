"""Pydantic models for governance and safety configuration."""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict


class GovernanceConfigRequest(BaseModel):
    """Payload for POST /governance/configure."""

    environment_name: str
    max_risk_threshold: float = 0.8
    compliance_hard_stop: bool = True
    human_in_the_loop: bool = False
    override_actions: Optional[Dict[str, str]] = None


class GovernanceConfig(BaseModel):
    """Governance configuration stored for an environment."""

    model_config = ConfigDict(extra="allow")

    environment_name: str
    max_risk_threshold: float = 0.8
    compliance_hard_stop: bool = True
    human_in_the_loop: bool = False
    safety_config: Optional[Dict] = None
