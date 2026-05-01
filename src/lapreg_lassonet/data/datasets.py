from __future__ import annotations

from pathlib import Path
from typing import Tuple, Any, Dict

import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from ..utils.paths import resolve_data_dir, find_project_root


def _load_sparse_or_dense(path_no_ext: Path):
    """
    Support:
      - Dense: <path>.npy
      - Sparse: <path>.npz  (scipy sparse)
    """
    npy_path = path_no_ext.with_suffix(".npy")
    npz_path = path_no_ext.with_suffix(".npz")
    if npz_path.exists():
        if sp is None:
            raise RuntimeError("scipy not installed, cannot load .npz sparse matrix.")
        return sp.load_npz(str(npz_path))
    if npy_path.exists():
        return np.load(str(npy_path))
    raise FileNotFoundError(f"Neither {npy_path} nor {npz_path} exists.")


def load_npy_dataset(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """
    Always resolves `data_dir` against project root so relative paths never break.

    Expect:
      <data_dir>/X.npy
      <data_dir>/y.npy
      <data_dir>/L.npy  OR <data_dir>/L.npz
    """
    project_root = find_project_root()
    data_path = resolve_data_dir(data_dir, project_root=project_root)

    X_path = data_path / "X.npy"
    y_path = data_path / "y.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing X.npy or y.npy.\n"
            f"  cwd: {Path.cwd()}\n"
            f"  project_root: {project_root}\n"
            f"  resolved_data_dir: {data_path}\n"
            f"  expected: {X_path} and {y_path}"
        )

    X = np.load(str(X_path))
    y = np.load(str(y_path))
    L = _load_sparse_or_dense(data_path / "L")

    meta = {
        "cwd": str(Path.cwd()),
        "project_root": str(project_root),
        "resolved_data_dir": str(data_path),
        "X_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
        "L_type": type(L).__name__,
    }
    return X, y, L, meta



# ====== placeholders for future real datasets ======
def load_mice_protein_placeholder(*args, **kwargs):
    """
    Placeholder:
    Later you will implement true loading for Mice Protein dataset.
    For now, keep pipeline working using npy dataset.
    """
    raise NotImplementedError("Implement load_mice_protein when you are ready.")
