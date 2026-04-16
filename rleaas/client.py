"""Base HTTP client for the Release (RLEaaS) SDK.

Provides both synchronous and asynchronous variants backed by httpx.
All domain-specific sub-clients (Environment, TrainingJob, Evaluation, …)
are attached as attributes of the root Client.

Usage::

    import rleaas

    # API key auto-read from RLEAAS_API_KEY environment variable
    client = rleaas.Client()

    # Or pass explicitly
    client = rleaas.Client(api_key="rleaas_sk_your_key_here")

    status = client.ping()
    # {'status': 'ok', 'version': '1.0.0'}
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from rleaas.exceptions import (
    RLEaaSAPIError,
    AuthenticationError,
    EnvironmentNotFound,
    QuotaExceededError,
    RateLimitError,
    TrainingJobNotFound,
    VerifierNotFound,
)


from rleaas.environments import EnvironmentsClient
from rleaas.tools import ToolsClient
from rleaas.training import TrainingClient
from rleaas.agents import AgentsClient
from rleaas.verifiers import VerifierClient
from rleaas.scenarios import ScenarioClient, ScenarioSuiteClient
from rleaas.evaluation import EvaluationClient
from rleaas.metrics import MetricsClient
from rleaas.audit import AuditLogClient
from rleaas.keys import KeysClient

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 30.0   # seconds
_DEFAULT_RETRIES = 3
_RETRY_BACKOFF = 1.5      # multiplied each attempt


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP error codes to typed SDK exceptions."""
    code = response.status_code
    if code < 400:
        return

    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text

    if code == 401:
        raise AuthenticationError(
            "Invalid or missing API key. Set RLEAAS_API_KEY or pass api_key= to Client()."
        )
    if code == 402:
        raise QuotaExceededError(detail)
    if code == 404:
        low = detail.lower()
        if "environment" in low:
            raise EnvironmentNotFound(detail)
        if "training" in low or "job" in low:
            raise TrainingJobNotFound(detail)
        if "verifier" in low:
            raise VerifierNotFound(detail)
        raise RLEaaSAPIError(404, detail)
    if code == 429:
        raise RateLimitError(detail)
    raise RLEaaSAPIError(code, detail)


class Client:
    """Synchronous Release SDK client.

    Parameters
    ----------
    api_key:
        Release API key. If omitted, reads ``RLEAAS_API_KEY`` from the
        environment. Obtain your key from the Release dashboard under
        Settings → API Keys.
    base_url:
        Override the API base URL (default ``http://localhost:8000``).
        In production this points to ``https://api.rleaas.centific.com``.
    timeout:
        Per-request timeout in seconds (default 30).
    retries:
        Number of retries on transient network errors (default 3).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        retries: int = _DEFAULT_RETRIES,
    ) -> None:
        resolved_key = api_key or os.environ.get("RLEAAS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._api_key = resolved_key
        self._timeout = timeout
        self._retries = retries
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._build_headers(),
        )

        # Sub-clients — mirroring the guide's object pattern
        self.Environment = EnvironmentsClient(self)
        self.Tools = ToolsClient(self)
        self.TrainingJob = TrainingClient(self)
        self.Agent = AgentsClient(self)
        self.Verifier = VerifierClient(self)
        self.Scenario = ScenarioClient(self)
        self.ScenarioSuite = ScenarioSuiteClient(self)
        self.Evaluation = EvaluationClient(self)
        self.Metrics = MetricsClient(self)
        self.AuditLog = AuditLogClient(self)
        self.Keys = KeysClient(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a request with retry logic; returns parsed JSON."""
        delay = _RETRY_BACKOFF
        last_exc: Exception = RuntimeError("No attempt made")
        for attempt in range(self._retries):
            try:
                response = self._http.request(method, path, params=params, json=json, data=data)
                _raise_for_status(response)
                return response.json()
            except (RLEaaSAPIError, AuthenticationError, QuotaExceededError,
                    RateLimitError, EnvironmentNotFound, TrainingJobNotFound,
                    VerifierNotFound):
                raise                       # typed — never retry
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
            if attempt < self._retries - 1:
                time.sleep(delay)
                delay *= _RETRY_BACKOFF
        raise RLEaaSAPIError(0, f"Request failed after {self._retries} attempts: {last_exc}")

    # ------------------------------------------------------------------
    # Public low-level HTTP methods (used by sub-clients)
    # ------------------------------------------------------------------

    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, *, json: Optional[Any] = None, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("POST", path, json=json, data=data, params=params)

    def put(self, path: str, *, json: Optional[Any] = None) -> Any:
        return self._request("PUT", path, json=json)

    def patch(self, path: str, *, json: Optional[Any] = None) -> Any:
        return self._request("PATCH", path, json=json)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # ------------------------------------------------------------------
    # Platform methods
    # ------------------------------------------------------------------

    def ping(self) -> Dict[str, Any]:
        """Verify connectivity and authentication.

        Returns a dict such as ``{'status': 'ok', 'version': '1.0.0'}``.
        Raises :class:`~rleaas.exceptions.AuthenticationError` if the API
        key is invalid.
        """
        return self._request("GET", "/api")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncClient:
    """Asynchronous Release SDK client.

    Drop-in async equivalent of :class:`Client`. Use inside ``async``
    functions or async frameworks (FastAPI, asyncio scripts, etc.).

    Example::

        async with rleaas.AsyncClient() as client:
            status = await client.ping()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        retries: int = _DEFAULT_RETRIES,
    ) -> None:
        resolved_key = api_key or os.environ.get("RLEAAS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._api_key = resolved_key
        self._timeout = timeout
        self._retries = retries
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._build_headers(),
        )

        # Sub-clients (async versions delegate to this client's async _request)
        self.Environment = EnvironmentsClient(self)    # type: ignore[assignment]
        self.Tools = ToolsClient(self)                 # type: ignore[assignment]
        self.TrainingJob = TrainingClient(self)         # type: ignore[assignment]
        self.Agent = AgentsClient(self)                # type: ignore[assignment]
        self.Verifier = VerifierClient(self)            # type: ignore[assignment]
        self.Scenario = ScenarioClient(self)            # type: ignore[assignment]
        self.ScenarioSuite = ScenarioSuiteClient(self)  # type: ignore[assignment]
        self.Evaluation = EvaluationClient(self)        # type: ignore[assignment]
        self.Metrics = MetricsClient(self)              # type: ignore[assignment]
        self.AuditLog = AuditLogClient(self)            # type: ignore[assignment]
        self.Keys = KeysClient(self)                    # type: ignore[assignment]

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
    ) -> Any:
        import asyncio

        delay = _RETRY_BACKOFF
        last_exc: Exception = RuntimeError("No attempt made")
        for attempt in range(self._retries):
            try:
                response = await self._http.request(method, path, params=params, json=json)
                _raise_for_status(response)
                return response.json()
            except (RLEaaSAPIError, AuthenticationError, QuotaExceededError,
                    RateLimitError, EnvironmentNotFound, TrainingJobNotFound,
                    VerifierNotFound):
                raise
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
            if attempt < self._retries - 1:
                await asyncio.sleep(delay)
                delay *= _RETRY_BACKOFF
        raise RLEaaSAPIError(0, f"Request failed after {self._retries} attempts: {last_exc}")

    async def get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, *, json: Optional[Any] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        return await self._request("POST", path, json=json, params=params)

    async def put(self, path: str, *, json: Optional[Any] = None) -> Any:
        return await self._request("PUT", path, json=json)

    async def patch(self, path: str, *, json: Optional[Any] = None) -> Any:
        return await self._request("PATCH", path, json=json)

    async def delete(self, path: str) -> Any:
        return await self._request("DELETE", path)

    async def ping(self) -> Dict[str, Any]:
        """Verify connectivity and authentication."""
        return await self._request("GET", "/api")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
