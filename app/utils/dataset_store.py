from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"


def _ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_summary() -> Dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    try:
        with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_dataset_summary(summary: Dict[str, Any]) -> None:
    _ensure_data_dirs()
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True)


def save_dataset_file(filename: str, content: bytes) -> str:
    _ensure_data_dirs()
    safe_name = Path(filename).name if filename else "dataset.csv"
    destination = UPLOAD_DIR / safe_name
    destination.write_bytes(content)
    return str(destination)
