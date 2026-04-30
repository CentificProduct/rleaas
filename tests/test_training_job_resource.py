from __future__ import annotations

from typing import Any, Dict, List

import pytest

from rleaas.exceptions import TrainingJobFailed
from rleaas.training import TrainingJobResource


class DummyClient:
    def __init__(self, get_responses: List[Any] | None = None) -> None:
        self._get_responses = list(get_responses or [])
        self.get_calls: List[Dict[str, Any]] = []

    def get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        self.get_calls.append({"path": path, "params": params})
        if self._get_responses:
            return self._get_responses.pop(0)
        return {}

    def post(self, path: str, json: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {"path": path, "json": json}


def test_wait_until_complete_raises_meaningful_failure() -> None:
    client = DummyClient(
        get_responses=[
            {"job_id": "job_1", "status": "failed", "error": "OOM on worker node"},
        ]
    )
    job = TrainingJobResource({"job_id": "job_1", "status": "running"}, client)  # type: ignore[arg-type]

    with pytest.raises(TrainingJobFailed, match="OOM on worker node"):
        job.wait_until_complete(timeout=1, poll_interval=0)


def test_get_logs_reads_common_payload_shape() -> None:
    client = DummyClient(get_responses=[{"logs": ["line1", "line2"]}])
    job = TrainingJobResource({"job_id": "job_2", "status": "running"}, client)  # type: ignore[arg-type]

    assert job.get_logs() == ["line1", "line2"]
    assert client.get_calls[0]["path"] == "/api/training/jobs/job_2/logs"


def test_stream_logs_yields_new_lines_and_completes() -> None:
    client = DummyClient(
        get_responses=[
            {"logs": ["first"]},
            {"job_id": "job_3", "status": "running"},
            {"logs": ["first", "second"]},
            {"job_id": "job_3", "status": "completed"},
            {"logs": ["first", "second"]},
        ]
    )
    job = TrainingJobResource({"job_id": "job_3", "status": "running"}, client)  # type: ignore[arg-type]

    assert list(job.stream_logs(poll_interval=0, timeout=1)) == ["first", "second"]
