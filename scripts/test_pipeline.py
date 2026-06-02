"""
scripts/test_pipeline.py - CLI tool for direct document intelligence testing.

Usage:
    python -m scripts.test_pipeline <path_to_document>

This script bypasses the API, Celery, and MinIO to run the analyzer 
directly on a local file. Ideal for debugging extraction logic.
"""

import sys
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.ocr_pipeline.core.smart_pipeline import SmartDocumentPipeline
from app.config import get_settings
from app.utils.logging import configure_logging, get_logger

def run_test(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    # 1. Setup
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    trace_id = str(uuid.uuid4())
    logger = get_logger(trace_id).bind(stage="cli_test")

    logger.info("starting_direct_pipeline_test", file=str(path))

    # 2. Run Pipeline
    try:
        pipeline = SmartDocumentPipeline(settings)
        result = pipeline.process(path, trace_id=trace_id)
        
        # 3. Display Results
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUCCESSFUL")
        print("="*50)
        print(f"Document ID: {result.document_id}")
        print(f"Status:      {result.status}")
        print(f"Duration:    {result.processing_time_ms}ms")
        print(f"Doc Type:    {result.final_export.document_type}")
        print("-" * 50)
        
        # Show structured data
        print("STRUCTURED DATA:")
        print(json.dumps(result.final_export.clean_dict() if hasattr(result.final_export, 'clean_dict') else asdict(result.final_export), indent=2, ensure_ascii=False))
        
        if result.warnings:
            print("-" * 50)
            print("WARNINGS:")
            for w in result.warnings:
                print(f"  - {w}")
        print("="*50 + "\n")

    except Exception as e:
        logger.exception("pipeline_test_failed", error=str(e))
        print(f"\nError during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_pipeline <path_to_document>")
        sys.exit(1)
    
    run_test(sys.argv[1])
