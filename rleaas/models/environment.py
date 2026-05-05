"""Pydantic models for environment-related API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

CreateEnvironmentSdk = Literal["gradio", "docker", "static", "custom"]

CreateEnvironmentLicense = Literal[
    "",
    "apache-2.0",
    "mit",
    "gpl-3.0",
    "bsd-3-clause",
    "proprietary",
    "other",
]

CreateEnvironmentCompute = Literal["cpu-basic", "cpu-upgrade", "gpu-t4", "gpu-a10"]

# Matches the UI dropdown in Step 1 — Basics (loaded from domains.json).
# id → UI label shown in the dropdown
#   "finance"         → 💰 fin-sim
#   "healthcare"      → 🏥 med-sim
#   "enterprise"      → 📋 dev-sim  (Jira / project workflows)
#   "human_resources" → 👥 hr-sim   (Workday / SAP / ADP)
CreateEnvironmentDomain = Literal[
    "finance",
    "healthcare",
    "enterprise",
    "human_resources",
]

# Maps UI domain id → SDK vertical name (and vice-versa for reference).
DOMAIN_TO_VERTICAL: Dict[str, str] = {
    "finance":         "FinSim",
    "healthcare":      "MedSim",
    "enterprise":      "DevSim",
    "human_resources": "HRSim",
}

VERTICAL_TO_DOMAIN: Dict[str, str] = {v: k for k, v in DOMAIN_TO_VERTICAL.items()}


class Environment(BaseModel):
    """A registered RL environment from the catalog or user-created."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    system: Optional[str] = None
    domain: Optional[str] = None
    multi_agent: bool = False
    observation_space: Optional[Dict[str, Any]] = None
    action_space: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class CreateEnvironmentRequest(BaseModel):
    """Payload for POST /api/custom-environments.

    ``domain`` and ``vertical`` serve the same conceptual purpose but use
    different value spaces:

    * ``domain`` — UI dropdown value (``"finance"``, ``"healthcare"``,
      ``"enterprise"``, ``"human_resources"``).  Use this when building
      tooling that mirrors the onboarding wizard.
    * ``vertical`` — catalog/training name (``"FinSim"``, ``"MedSim"``,
      ``"DevSim"``, ``"HRSim"``).  Use this when working with training jobs
      or verifier registries.

    You may supply both; the server stores whichever field it reads.
    :data:`DOMAIN_TO_VERTICAL` and :data:`VERTICAL_TO_DOMAIN` can be used
    to convert between the two.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    owner: str = Field(default="")
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    description: str = Field(default="", max_length=200)
    license: CreateEnvironmentLicense = ""
    sdk: CreateEnvironmentSdk = "gradio"
    compute: CreateEnvironmentCompute = "cpu-basic"
    domain: Optional[CreateEnvironmentDomain] = None
    vertical: Optional[str] = None
    source: str = "custom"


class UpdateEnvironmentSystemRequest(BaseModel):
    """Payload for PUT /api/environments/{name}/system."""

    system: str
    metadata: Optional[Dict[str, Any]] = None


class Scenario(BaseModel):
    """A scenario attached to an environment."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    environment_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
