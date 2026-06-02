"""
Logging and terminal-output helpers for the OCR pipeline.
"""

from __future__ import annotations
from contextlib import contextmanager
import os
import sys
from typing import Iterator

import structlog
from app.utils.logging import get_logger as get_trace_logger


def get_logger(name: str | None = None):
    """
    Backward-compatible logger getter for the OCR pipeline.

    IMPORTANT: trace_id and stage are injected via structlog contextvars
    from app.utils.logging (API/Celery set trace_id; pipeline stage wrappers set stage).
    """

    return structlog.get_logger(name)


@contextmanager
def suppress_external_output(enabled: bool = True) -> Iterator[None]:
    """Silence noisy third-party native libraries during model calls."""
    if not enabled:
        yield
        return

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdout_fd: int | None = None
    old_stderr_fd: int | None = None
    devnull = None
    try:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            devnull = open(os.devnull, "w", encoding="utf-8")
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            sys.stdout = devnull
            sys.stderr = devnull
        except OSError:
            devnull = None
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if old_stdout_fd is not None:
            os.dup2(old_stdout_fd, 1)
            os.close(old_stdout_fd)
        if devnull is not None:
            devnull.close()


def terminal_text(value: object) -> str:
    """Return text that cannot crash legacy Windows terminal encodings."""
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
