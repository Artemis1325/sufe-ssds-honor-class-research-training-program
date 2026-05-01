from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_MARKERS = ["pyproject.toml", ".git", "src"]


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find repo/project root by walking upwards from `start`.
    A directory is considered root if it contains any marker:
      - pyproject.toml
      - .git
      - src
    """
    if start is None:
        start = Path(__file__).resolve()
    cur = start if start.is_dir() else start.parent

    for _ in range(30):  # safety bound
        if any((cur / m).exists() for m in _MARKERS):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # fallback: 2 levels up from this file (…/src/lapreg_lassonet/utils/paths.py)
    # -> …/src/lapreg_lassonet/utils -> …/src/lapreg_lassonet -> …/src -> …/
    return Path(__file__).resolve().parents[3]


def resolve_data_dir(data_dir: str, project_root: Optional[Path] = None) -> Path:
    """
    Convert a user-provided data_dir into an absolute, normalized path.

    Rules:
    - If data_dir is absolute: use it
    - Else: interpret it relative to project_root
    """
    if project_root is None:
        project_root = find_project_root()

    p = Path(data_dir)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def resolve_results_dir(results_dir: str, project_root: Optional[Path] = None) -> Path:
    if project_root is None:
        project_root = find_project_root()
    p = Path(results_dir)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()
