import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import os
import numpy as np

from lapreg_lassonet.data.graphs import corr_knn_graph, adjacency_to_laplacian
from lapreg_lassonet.utils.io import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="datasets")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--normalized", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # ====== TODO: replace with real dataset loading (e.g., Mice Protein) ======
    # For now, create a dummy placeholder if X/y not provided
    # You should replace this block with your actual datasets import.
    X_path = os.path.join(args.out_dir, "X.npy")
    y_path = os.path.join(args.out_dir, "y.npy")
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            "X.npy/y.npy not found. Put your dataset at datasets/X.npy,datasets/y.npy "
            "or modify this script to generate them."
        )

    X = np.load(X_path)
    y = np.load(y_path)

    A = corr_knn_graph(X, k=args.k, abs_corr=True)
    L = adjacency_to_laplacian(A, normalized=args.normalized)

    np.save(os.path.join(args.out_dir, "L.npy"), L)
    print("Saved L.npy to", args.out_dir)
    print("Shapes:", "X", X.shape, "y", y.shape, "L", L.shape)


if __name__ == "__main__":
    main()
