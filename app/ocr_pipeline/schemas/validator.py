"""Validate extracted fields against Pydantic schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


def validate(doc_type: str, fields: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Return validated fields and validation warnings.

    If pydantic is not installed or the schema is unknown, returns the original
    fields without warnings.
    """
    try:
        if doc_type == "invoice":
            from app.ocr_pipeline.schemas.invoice import InvoiceSchema

            model = InvoiceSchema.model_validate(fields, strict=False)
            return model.model_dump(exclude_none=True), []

        if doc_type == "cv":
            from app.ocr_pipeline.schemas.cv import CVSchema

            model = CVSchema.model_validate(fields, strict=False)
            return model.model_dump(exclude_none=True), []

    except ImportError:
        return fields, []
    except Exception as exc:
        LOGGER.warning("Schema validation failed for %s: %s", doc_type, exc)
        return fields, [str(exc)]

    return fields, []

