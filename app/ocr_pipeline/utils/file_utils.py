"""
File I/O and serialization utilities.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from dataclasses import asdict, is_dataclass
import numpy as np

def serialize(value: Any) -> Any:
    """Serialize nested dataclasses and numpy arrays into JSON-safe values."""
    if is_dataclass(value):
        return {key: serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def ensure_directory(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def save_json(data: Any, file_path: str | Path) -> None:
    """Save data as JSON to the specified path."""
    path = Path(file_path)
    ensure_directory(path.parent)
    serialized_data = serialize(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, ensure_ascii=False, indent=2)

def save_text(text: str, file_path: str | Path) -> None:
    """Save string to a text file."""
    path = Path(file_path)
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

