"""
Grounding / Hallucination checker.
Verifies that AI-extracted fields are present in the source OCR text.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set
import re

from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)

class GroundingChecker:
    """Verifies that extracted data is grounded in the source text."""

    def verify(self, extracted_fields: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        """
        Checks every string value in extracted_fields against source_text.
        Returns a report of verified vs unverified fields.
        """
        source_normalized = self._normalize(source_text)
        results = {}
        
        for field, value in self._flatten_dict(extracted_fields).items():
            if not isinstance(value, str) or len(value) < 3:
                results[field] = True # Skip small or non-string values
                continue
                
            val_normalized = self._normalize(value)
            # Check if normalized value is in source
            is_present = val_normalized in source_normalized
            
            # Fuzzy fallback for minor OCR noise
            if not is_present and len(val_normalized) > 10:
                is_present = self._fuzzy_check(val_normalized, source_normalized)
                
            results[field] = is_present
            
            if not is_present:
                LOGGER.warning("Field '%s' with value '%s' not grounded in OCR text", field, value)

        return {
            "is_grounded": all(results.values()),
            "field_status": results
        }

    def _normalize(self, text: str) -> str:
        # Remove non-alphanumeric and lowercase
        return re.sub(r'[^a-z0-9]', '', text.lower())

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, list):
                for i, val in enumerate(v):
                    if isinstance(val, dict):
                        items.extend(self._flatten_dict(val, f"{new_key}[{i}]").items())
                    else:
                        items.append((f"{new_key}[{i}]", val))
            else:
                items.append((new_key, v))
        return dict(items)

    def _fuzzy_check(self, val: str, source: str) -> bool:
        # Very basic fuzzy check: are most characters present in order?
        # This is a placeholder for Levenshtein or similar.
        return val[:10] in source or val[-10:] in source

