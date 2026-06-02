"""Shared boilerplate for all subprocess worker entry points."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


def run_worker(
    argv: list[str],
    expected_args: int,
    handler: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> int:
    """
    Standard worker entry point.

    argv            sys.argv including the script name at index 0
    expected_args   number of positional args after the script name
    handler         fn(payload_dict) -> result_dict
    """
    args = argv[1:]
    if len(args) != expected_args:
        LOGGER.error(
            "worker_usage_error",
            expected_args=expected_args,
            got_args=len(args),
            usage=f"{argv[0]} " + " ".join(f"<arg{i}>" for i in range(expected_args)),
        )
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("worker_input_read_failed", input_path=str(input_path), error=str(exc))
        output_path.write_text(
            json.dumps({"error": f"could not read input: {exc}"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1

    try:
        result = handler(payload)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        LOGGER.warning("worker_handler_failed", error=str(exc), error_type=type(exc).__name__)
        result = {"error": f"{type(exc).__name__}: {exc}"}

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0

