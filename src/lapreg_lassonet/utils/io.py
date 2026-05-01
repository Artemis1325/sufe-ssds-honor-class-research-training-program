from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
from pathlib import Path
from .paths import resolve_results_dir, find_project_root

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def results_subdir(results_dir: str, run_name: str) -> str:
    root = find_project_root()
    res = resolve_results_dir(results_dir, project_root=root)
    path = Path(res) / run_name
    ensure_dir(str(path))
    return str(path)
