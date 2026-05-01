from __future__ import annotations

from typing import Tuple, Optional, Union, Any, Dict
import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def adjacency_to_laplacian(A: np.ndarray, normalized: bool = False) -> np.ndarray:
    """
    Dense Laplacian from dense adjacency A.
    L = D - A (or normalized variant)
    """
    A = np.asarray(A, dtype=float)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    if not normalized:
        L = np.diag(d) - A
        return L
    # normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def corr_knn_graph(X: np.ndarray, k: int = 10, abs_corr: bool = True) -> np.ndarray:
    """
    Build a simple KNN graph over features using correlation as similarity.
    Returns dense adjacency A (p,p).
    Note: For large p this is expensive; for big datasets you will switch to sparse and/or external edgelist.
    """
    X = np.asarray(X)
    # corr among features: p x p
    C = np.corrcoef(X, rowvar=False)
    if abs_corr:
        C = np.abs(C)
    np.fill_diagonal(C, 0.0)
    p = C.shape[0]
    A = np.zeros((p, p), dtype=float)

    # connect top-k neighbors per node
    for i in range(p):
        idx = np.argpartition(-C[i], kth=min(k, p - 1))[:k]
        A[i, idx] = C[i, idx]
    # symmetrize
    A = np.maximum(A, A.T)
    return A


def sparse_adjacency_to_laplacian(A, normalized: bool = False):
    """
    Sparse Laplacian from sparse adjacency A.
    """
    if sp is None:
        raise RuntimeError("scipy not installed; cannot build sparse Laplacian.")

    A = A.tocsr().astype(float)
    A.setdiag(0.0)
    A.eliminate_zeros()

    d = np.asarray(A.sum(axis=1)).ravel()
    if not normalized:
        D = sp.diags(d, format="csr")
        return D - A

    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = sp.diags(d_inv_sqrt, format="csr")
    I = sp.eye(A.shape[0], format="csr")
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def pearson_knn_laplacian(
    X: np.ndarray,
    k: int = 10,
    abs_corr: bool = True,
    normalized: bool = False,
):
    """
    Build a sparse feature graph from training data only using Pearson correlation.

    Returns:
      - L: scipy sparse Laplacian (csr)
      - info: lightweight graph statistics
    """
    if sp is None:
        raise RuntimeError("scipy not installed; cannot build sparse Laplacian.")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape={X.shape}")

    n_samples, p = X.shape
    if p == 0:
        raise ValueError("X has zero features.")

    k_eff = max(1, min(int(k), max(1, p - 1)))

    X_centered = X - X.mean(axis=0, keepdims=True)
    if n_samples > 1:
        cov = (X_centered.T @ X_centered) / float(n_samples - 1)
    else:
        cov = np.zeros((p, p), dtype=np.float32)
    std = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    denom = np.outer(std, std)
    C = np.divide(cov, denom, out=np.zeros_like(cov, dtype=np.float32), where=denom > 0)
    if abs_corr:
        C = np.abs(C)
    np.fill_diagonal(C, 0.0)

    rows = []
    cols = []
    vals = []
    for i in range(p):
        idx = np.argpartition(-C[i], kth=k_eff - 1)[:k_eff]
        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
        vals.extend(C[i, idx].tolist())

    A = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(p, p),
    ).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()

    L = sparse_adjacency_to_laplacian(A, normalized=normalized)
    deg = np.asarray(A.sum(axis=1)).ravel()
    info: Dict[str, Any] = {
        "graph_mode": "pearson_train",
        "pearson_k": int(k_eff),
        "pearson_abs_corr": bool(abs_corr),
        "pearson_normalized_laplacian": bool(normalized),
        "n_samples_used": int(n_samples),
        "n_features": int(p),
        "adj_nnz": int(A.nnz),
        "undirected_edges": int(A.nnz // 2),
        "degree_min": float(deg.min()) if deg.size else 0.0,
        "degree_max": float(deg.max()) if deg.size else 0.0,
        "degree_mean": float(deg.mean()) if deg.size else 0.0,
        "isolated_nodes": int((deg == 0).sum()) if deg.size else 0,
    }
    return L, info


def edgelist_to_sparse_laplacian(
    num_nodes: int,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
    normalized: bool = False,
):
    """
    edges: (m,2) int
    weights: (m,) float optional
    returns scipy sparse L (csr)
    """
    if sp is None:
        raise RuntimeError("scipy not installed; cannot build sparse Laplacian.")

    edges = np.asarray(edges, dtype=int)
    if weights is None:
        weights = np.ones(edges.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    i = edges[:, 0]
    j = edges[:, 1]
    data = weights

    # undirected: add both directions
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([data, data])

    A = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()

    d = np.array(A.sum(axis=1)).ravel()
    if not normalized:
        D = sp.diags(d, format="csr")
        L = D - A
        return L

    # normalized Laplacian
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = sp.diags(d_inv_sqrt, format="csr")
    I = sp.eye(num_nodes, format="csr")
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L
