from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, roc_auc_score


def nnz(x: np.ndarray, eps: float = 1e-8) -> int:
    return int(np.sum(np.abs(x) > eps))


def lap_energy(theta: np.ndarray, L: Any) -> float:
    """
    Works for:
      - dense numpy L
      - scipy sparse L
    """
    theta = theta.astype(float)
    if hasattr(L, "dot"):  # numpy matrix or scipy sparse
        v = L.dot(theta)
    else:
        v = L @ theta
    return float(theta @ v)


def binary_metrics_from_logits(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    probs = expit(logits)
    preds = (probs >= threshold).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, preds)),
    }
    # AUC requires both classes present
    try:
        out["auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["auc"] = float("nan")
    return out


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)
