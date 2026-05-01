# src/prepare_dataset.py
"""
Prepare dataset + feature graph (W) and Laplacian (L).
Saves: datasets/X.npy, datasets/y.npy, datasets/W.npy, datasets/L.npy, datasets/feature_names.json

Two graph options:
 - corr_threshold: build W from Pearson corr of features, threshold small correlations
 - knn: build W by connecting each feature to its k nearest neighbors (in feature-feature space using corr)
Default: corr_threshold
"""
import os
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

os.makedirs("datasets", exist_ok=True)

def build_W_corr(X, threshold=0.2, tau=3.0):
    # X: (n_samples, p)
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 1.0)
    W = np.exp(tau * np.abs(corr)) - 1.0
    W[np.abs(corr) < threshold] = 0.0
    W = (W + W.T) / 2.0
    return W

def build_W_knn(X, k=5, mode='connectivity', metric='correlation'):
    # Build feature-feature kNN using transposed datasets
    # sklearn kneighbors_graph returns sparse adjacency for samples; we feed features as "samples"
    # metric 'correlation' works for sklearn >= 0.22
    Xf = X.T  # shape p x n_samples
    A = kneighbors_graph(Xf, n_neighbors=k, mode=mode, metric=metric, include_self=False)
    A = 0.5 * (A.toarray() + A.toarray().T)
    return A

def laplacian_from_W(W, normalized=True):
    d = np.sum(W, axis=1)
    if normalized:
        with np.errstate(divide='ignore'):
            d_sqrt_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_sqrt_inv)
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        D = np.diag(d)
        L = D - W
    return L

def main(graph_type="corr", corr_threshold=0.2, tau=3.0, knn_k=5):
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)
    feature_names = [str(x) for x in data.feature_names]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if graph_type == "corr":
        W = build_W_corr(Xs, threshold=corr_threshold, tau=tau)
    else:
        W = build_W_knn(Xs, k=knn_k)

    L = laplacian_from_W(W, normalized=True)

    np.save("datasets/X.npy", Xs)
    np.save("datasets/y.npy", y)
    np.save("datasets/W.npy", W)
    np.save("datasets/L.npy", L)

    with open("datasets/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print("Saved: datasets/X.npy, datasets/y.npy, datasets/W.npy, datasets/L.npy, datasets/feature_names.json")
    print("W shape:", W.shape, "L shape:", L.shape)

if __name__ == "__main__":
    # default: corr-based W
    main(graph_type="corr", corr_threshold=0.2, tau=3.0, knn_k=5)
