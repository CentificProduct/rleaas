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


class Environment(BaseModel):
    """A registered RL environment from the catalog or user-created."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    system: Optional[str] = None
    multi_agent: bool = False
    observation_space: Optional[Dict[str, Any]] = None
    action_space: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class CreateEnvironmentRequest(BaseModel):
    """Payload for POST /api/environments."""

    model_config = ConfigDict(str_strip_whitespace=True)

    owner: str = Field(default="")
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    description: str = Field(default="", max_length=200)
    license: CreateEnvironmentLicense = ""
    sdk: CreateEnvironmentSdk = "gradio"
    compute: CreateEnvironmentCompute = "cpu-basic"


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
