"""Release SDK — RL Environments as a Service by Centific.

Quick start::

    import rleaas

    # Reads RLEAAS_API_KEY from environment automatically
    client = rleaas.Client()

    status = client.ping()
    # {'status': 'ok', 'version': '1.0.0'}
"""

from rleaas.client import Client, AsyncClient
from rleaas.environments import EnvironmentsClient, EnvironmentResource
from rleaas.tools import ToolsClient
from rleaas.training import TrainingClient, TrainingJobResource
from rleaas.agents import AgentsClient, AgentResource
from rleaas.verifiers import VerifierClient, VerifierResource
from rleaas.scenarios import ScenarioClient, ScenarioSuiteClient
from rleaas.evaluation import EvaluationClient, EvaluationJobResource
from rleaas.metrics import MetricsClient
from rleaas.audit import AuditLogClient
from rleaas.exceptions import (
    RLEaaSError,
    RLEaaSAPIError,
    AuthenticationError,
    QuotaExceededError,
    RateLimitError,
    EnvironmentNotFound,
    TrainingJobNotFound,
    VerifierNotFound,
)
from rleaas.models import (
    # environment
    Environment,
    CreateEnvironmentRequest,
    UpdateEnvironmentSystemRequest,
    Scenario,
    # training
    TrainingRequest,
    TrainingResponse,
    TrainingJob,
    TrainingConfig,
    Agent,
    # rollout / evaluation
    Rollout,
    RolloutStep,
    RolloutComparison,
    TimelineEvent,
    VerifierResult,
    KPI,
    HumanEvalRequest,
    HumanEvalStepScore,
    # verifier
    Verifier,
    VerifierConfigRequest,
    VerifierListResponse,
    FailurePolicy,
    # governance
    GovernanceConfigRequest,
    GovernanceConfig,
)

__version__ = "1.0.1"

__all__ = [
    # clients
    "Client",
    "AsyncClient",
    # sub-clients
    "EnvironmentsClient",
    "EnvironmentResource",
    "ToolsClient",
    "TrainingClient",
    "TrainingJobResource",
    "AgentsClient",
    "AgentResource",
    "VerifierClient",
    "VerifierResource",
    "ScenarioClient",
    "ScenarioSuiteClient",
    "EvaluationClient",
    "EvaluationJobResource",
    "MetricsClient",
    "AuditLogClient",
    # exceptions
    "RLEaaSError",
    "RLEaaSAPIError",
    "AuthenticationError",
    "QuotaExceededError",
    "RateLimitError",
    "EnvironmentNotFound",
    "TrainingJobNotFound",
    "VerifierNotFound",
    # models — environment
    "Environment",
    "CreateEnvironmentRequest",
    "UpdateEnvironmentSystemRequest",
    "Scenario",
    # models — training
    "TrainingRequest",
    "TrainingResponse",
    "TrainingJob",
    "TrainingConfig",
    "Agent",
    # models — rollout / evaluation
    "Rollout",
    "RolloutStep",
    "RolloutComparison",
    "TimelineEvent",
    "VerifierResult",
    "KPI",
    "HumanEvalRequest",
    "HumanEvalStepScore",
    # models — verifier
    "Verifier",
    "VerifierConfigRequest",
    "VerifierListResponse",
    "FailurePolicy",
    # models — governance
    "GovernanceConfigRequest",
    "GovernanceConfig",
]
