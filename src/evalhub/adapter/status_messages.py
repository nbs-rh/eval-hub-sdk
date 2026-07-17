"""Sanitize consumer-facing status error/warning messages.

Adapters often surface raw exceptions from model endpoints, MLflow, OCI, or
eval-hub itself. Those strings can include URLs or hostnames that should not be
persisted or shown to API consumers. This module shortens such messages while
leaving enough context to be useful (for example HTTP status codes).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from .models.job import JobStatusUpdate, MessageInfo

logger = logging.getLogger(__name__)

# Matches adapter errors like:
# "Model endpoint returned HTTP 404: 404 Client Error: Not Found for url: http://..."
# Capture group 1 is the consumer-safe prefix (through the HTTP status code).
_ENDPOINT_HTTP_DETAIL = re.compile(
    r"(?i)^(.+\bendpoint returned HTTP \d+)\s*:\s*\S"
)

# Absolute URLs — replaced with path (+ query/fragment) so the message stays useful.
# Optional leading preposition is dropped when the URL has no useful path.
_PREP_URL = re.compile(
    r"(?i)(?:(?P<prep>\s+(?:at|from|to|on|via))\s+)?"
    r"(?P<url>https?://[^\s\"'<>]+)"
)

# Bare hostnames that commonly leak from sidecar / in-cluster errors (no scheme).
# Intentionally narrow to avoid stripping filenames like ``report.json``.
# Optional leading preposition is dropped when there is no path to keep.
_PREP_HOSTNAME = re.compile(
    r"(?i)(?:(?P<prep>\s+(?:at|from|to|on|via))\s+)?"
    r"\b(?:"
    r"localhost(?::\d+)?"
    r"|127\.0\.0\.1(?::\d+)?"
    r"|(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+svc\.cluster\.local(?::\d+)?"
    r")(?P<path>/[^\s\"'<>]*)?"
)

# Safety net for a dangling preposition left at the end of the message.
_TRAILING_PREP = re.compile(r"(?i)\s+\b(?:at|from|to|on|via)\b\s*$")

_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")


def _url_path(url: str) -> str:
    """Return path (+ query/fragment) for ``url``, or ``\"\"`` if none is useful."""
    parsed = urlparse(url)
    path = parsed.path or ""
    if path in ("", "/") and not parsed.query and not parsed.fragment:
        return ""
    if not path.startswith("/"):
        path = f"/{path}"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    if parsed.fragment:
        path = f"{path}#{parsed.fragment}"
    return path


def _prep_url_to_path(match: re.Match[str]) -> str:
    """Replace an absolute URL with its path; drop a leading prep if nothing remains."""
    path = _url_path(match.group("url"))
    if not path:
        return ""
    prep = match.group("prep") or ""
    return f"{prep} {path}" if prep else path


def _prep_hostname_to_path(match: re.Match[str]) -> str:
    """Drop a bare hostname, keeping path and a leading prep only when a path remains."""
    path = match.group("path") or ""
    if not path:
        return ""
    prep = match.group("prep") or ""
    return f"{prep} {path}" if prep else path


def sanitize_consumer_message(message: str) -> str:
    """Return a consumer-safe message with URLs/hostnames removed when present.

    Prefer truncating well-known ``endpoint returned HTTP <code>: <detail>``
    errors to the status-code prefix (same behavior as eval-hub persistence).
    Otherwise replace absolute URLs with their path (e.g.
    ``https://host/api/v1/jobs`` → ``/api/v1/jobs``), strip bare hostnames while
    keeping any path, and drop dangling prepositions like ``at`` /
    ``from`` when the host/URL is removed entirely.

    The full original message is logged at INFO before any shortening.
    Messages that do not contain URL/hostname detail are returned unchanged
    (and are not logged by this helper).
    """
    if not message:
        return message

    match = _ENDPOINT_HTTP_DETAIL.match(message)
    if match:
        shortened = match.group(1).strip()
        if shortened != message:
            logger.info(
                "Sanitized status message for consumer "
                "(full message logged before shortening): %s",
                message,
            )
        return shortened

    cleaned = _PREP_URL.sub(_prep_url_to_path, message)
    cleaned = _PREP_HOSTNAME.sub(_prep_hostname_to_path, cleaned)
    cleaned = _TRAILING_PREP.sub("", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = cleaned.strip(" \t,;:")

    if cleaned and cleaned != message:
        logger.info(
            "Sanitized status message for consumer "
            "(full message logged before shortening): %s",
            message,
        )
        return cleaned

    return message


def _sanitize_message_info(info: MessageInfo | None) -> MessageInfo | None:
    if info is None:
        return None
    sanitized = sanitize_consumer_message(info.message)
    if sanitized == info.message:
        return info
    return info.model_copy(update={"message": sanitized})


def sanitize_status_update(update: JobStatusUpdate) -> JobStatusUpdate:
    """Return a copy of ``update`` with consumer-safe error/warning messages."""
    error_message = _sanitize_message_info(update.error_message)
    warning_message = _sanitize_message_info(update.warning_message)

    if (
        error_message is update.error_message
        and warning_message is update.warning_message
    ):
        return update

    return update.model_copy(
        update={
            "error_message": error_message,
            "warning_message": warning_message,
        }
    )
