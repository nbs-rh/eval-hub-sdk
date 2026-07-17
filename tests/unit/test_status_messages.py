"""Tests for consumer-facing status message sanitization."""

from __future__ import annotations

import logging

import pytest

from evalhub.adapter.models.job import JobStatusUpdate, MessageInfo
from evalhub.adapter.status_messages import (
    sanitize_consumer_message,
    sanitize_status_update,
)
from evalhub.models.api import JobStatus


@pytest.mark.parametrize(
    ("message", "want"),
    [
        (
            "Model endpoint returned HTTP 404: 404 Client Error: Not Found for url: "
            "http://localhost:8080/v1/completions",
            "Model endpoint returned HTTP 404",
        ),
        (
            "MLflow endpoint returned HTTP 502: 502 Server Error: Bad Gateway for url: "
            "http://localhost:8080/api/2.0/mlflow/runs/create",
            "MLflow endpoint returned HTTP 502",
        ),
        (
            "OCI endpoint returned HTTP 401: unauthorized for url: "
            "https://quay.io/v2/evalhub/results/blobs/upload/",
            "OCI endpoint returned HTTP 401",
        ),
        (
            "Model endpoint returned HTTP 404",
            "Model endpoint returned HTTP 404",
        ),
        (
            "Connection failed: timeout talking to sidecar",
            "Connection failed: timeout talking to sidecar",
        ),
        ("", ""),
        (
            "Upload failed for url: http://registry.svc.cluster.local:5000/v2/",
            "Upload failed for url: /v2/",
        ),
        (
            "Could not reach https://evalhub.example.com/api/v1/jobs",
            "Could not reach /api/v1/jobs",
        ),
        (
            "Could not reach https://evalhub.example.com/api/v1/jobs?limit=10",
            "Could not reach /api/v1/jobs?limit=10",
        ),
        (
            "Sidecar unavailable at localhost:8080",
            "Sidecar unavailable",
        ),
        (
            "Sidecar unavailable at https://localhost:8080",
            "Sidecar unavailable",
        ),
        (
            "Request failed at localhost:8080/v1/completions",
            "Request failed at /v1/completions",
        ),
        (
            "Failed writing report.json to results dir",
            "Failed writing report.json to results dir",
        ),
    ],
)
def test_sanitize_consumer_message(message: str, want: str) -> None:
    assert sanitize_consumer_message(message) == want


def test_sanitize_consumer_message_logs_full_message_before_shortening(
    caplog: pytest.LogCaptureFixture,
) -> None:
    full = (
        "Model endpoint returned HTTP 500: upstream boom for url: "
        "http://localhost:8080/v1/completions"
    )
    with caplog.at_level(logging.INFO, logger="evalhub.adapter.status_messages"):
        assert sanitize_consumer_message(full) == "Model endpoint returned HTTP 500"

    assert full in caplog.text


def test_sanitize_consumer_message_does_not_log_when_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = "Model endpoint returned HTTP 404"
    with caplog.at_level(logging.INFO, logger="evalhub.adapter.status_messages"):
        assert sanitize_consumer_message(message) == message

    assert "Sanitized status message" not in caplog.text


def test_sanitize_status_update_sanitizes_error_and_warning() -> None:
    update = JobStatusUpdate(
        status=JobStatus.FAILED,
        error_message=MessageInfo(
            message=(
                "Model endpoint returned HTTP 404: 404 Client Error: Not Found "
                "for url: http://localhost:8080/v1/completions"
            ),
            message_code="evaluation_error",
        ),
        warning_message=MessageInfo(
            message="Slow response from https://judge.example.com/v1",
            message_code="slow_response",
        ),
    )

    sanitized = sanitize_status_update(update)

    assert sanitized is not update
    assert sanitized.error_message is not None
    assert sanitized.error_message.message == "Model endpoint returned HTTP 404"
    assert sanitized.error_message.message_code == "evaluation_error"
    assert sanitized.warning_message is not None
    assert sanitized.warning_message.message == "Slow response from /v1"
    assert update.error_message is not None
    assert "localhost:8080" in update.error_message.message


def test_sanitize_status_update_noop_when_clean() -> None:
    update = JobStatusUpdate(
        status=JobStatus.FAILED,
        error_message=MessageInfo(
            message="evaluation failed",
            message_code="evaluation_error",
        ),
    )

    assert sanitize_status_update(update) is update
