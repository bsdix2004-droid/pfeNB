"""
app/utils/logging.py

Structured JSON logging for Scanalyze (FastAPI + Celery + OCR pipeline).

Requirements implemented:
- structlog JSON logs
- trace_id generated at API layer and propagated through Celery/engine/pipeline
- get_logger(trace_id) factory

Every log entry includes at least:
  - trace_id (string or null)
  - stage (string or null)
  - level
  - event
Optionally:
  - duration_ms (int) for timed operations
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars, unbind_contextvars


def _ensure_defaults(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    event_dict.setdefault("trace_id", None)
    event_dict.setdefault("stage", "unknown")
    # duration_ms is only set where applicable; keep key present only when provided.
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure stdlib logging + structlog to emit JSON to stderr.

    Call once during process startup (FastAPI app startup, Celery worker bootstrap,
    and any CLI entrypoints).
    """

    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    # Stdlib config (used by dependencies); we route it through structlog too.
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            _ensure_defaults,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )


def get_logger(trace_id: str | uuid.UUID | None):
    """
    Logger factory requested by the project requirements.

    Usage:
        log = get_logger(trace_id).bind(stage="api")
        log.info("job_created", job_id=str(job.id))
    """

    if trace_id:
        bind_contextvars(trace_id=str(trace_id))
    return structlog.get_logger()


@contextmanager
def log_stage(trace_id: str | uuid.UUID | None, stage: str) -> Iterator[structlog.BoundLogger]:
    """
    Context manager that:
      - binds stage
      - measures duration_ms
      - emits stage_start / stage_end events
    """

    if stage:
        bind_contextvars(stage=stage)
    log = get_logger(trace_id)
    started = time.perf_counter()
    log.info("stage_start")
    try:
        yield log
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.info("stage_end", duration_ms=duration_ms)
        # Do not leak stage into subsequent logs in the same worker thread.
        try:
            unbind_contextvars("stage")
        except Exception:
            pass
