"""SDK-specific exceptions for the Release (RLEaaS) client."""


class RLEaaSError(Exception):
    """Base exception for all Release SDK errors."""


class RLEaaSAPIError(RLEaaSError):
    """HTTP error returned by the Release API."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class EnvironmentNotFound(RLEaaSError):
    """Raised when an environment ID or name does not exist."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(f"Environment not found: {ref!r}")


class TrainingJobNotFound(RLEaaSError):
    """Raised when a training job ID cannot be located."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Training job not found: {job_id!r}")


class VerifierNotFound(RLEaaSError):
    """Raised when a verifier ID cannot be located."""

    def __init__(self, verifier_id: str) -> None:
        self.verifier_id = verifier_id
        super().__init__(f"Verifier not found: {verifier_id!r}")


class AuthenticationError(RLEaaSError):
    """Raised when a request is rejected due to missing or invalid API key."""


class QuotaExceededError(RLEaaSError):
    """Raised when compute budget or usage quota is reached (HTTP 402)."""


class RateLimitError(RLEaaSError):
    """Raised when too many API calls are made in a short period (HTTP 429)."""