import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_adjacency(data_dir: Path):
    a_npz_path = data_dir / "A.npz"
    l_npz_path = data_dir / "L.npz"
    w_npy_path = data_dir / "W.npy"
    l_npy_path = data_dir / "L.npy"

    if a_npz_path.exists():
        A = sparse.load_npz(a_npz_path).tocsr()
        A = A.copy()
        A.setdiag(0)
        A.eliminate_zeros()
        A.data[:] = 1
        return A, "A.npz"

    elif l_npz_path.exists():
        L = sparse.load_npz(l_npz_path).tocsr()
        A = L.copy()
        A.setdiag(0)
        A.data = (np.abs(A.data) > 0).astype(np.int8)
        A.eliminate_zeros()
        return A, "L.npz"

    elif w_npy_path.exists():
        W = np.load(w_npy_path)
        A = (W != 0).astype(np.int8)
        np.fill_diagonal(A, 0)
        return sparse.csr_matrix(A), "W.npy"

    elif l_npy_path.exists():
        L = np.load(l_npy_path)
        A = (np.abs(L) > 0).astype(np.int8)
        np.fill_diagonal(A, 0)
        return sparse.csr_matrix(A), "L.npy"

    else:
        raise FileNotFoundError(
            f"在 {data_dir} 下找不到 A.npz / L.npz / W.npy / L.npy"
        )

def parse_selected(arr, n_features):
    arr = np.asarray(arr)

    # 情况1：布尔 mask
    if arr.dtype == bool and arr.ndim == 1 and arr.shape[0] == n_features:
        idx = np.flatnonzero(arr)
        return idx

    # 情况2：0/1 mask
    if arr.ndim == 1 and arr.shape[0] == n_features and np.all(np.isin(arr, [0, 1])):
        idx = np.flatnonzero(arr)
        return idx

    # 情况3：直接存的是 index 列表
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        if arr.size == 0:
            return np.array([], dtype=int)
        if arr.max() < n_features and arr.min() >= 0:
            return np.unique(arr.astype(int))

    raise ValueError(
        f"无法识别 selected 文件格式: shape={arr.shape}, dtype={arr.dtype}, n_features={n_features}"
    )


def compute_metrics(A_csr, selected_idx):
    n_selected = int(len(selected_idx))

    if n_selected == 0:
        return {
            "n_selected": 0,
            "n_edges_induced": 0,
            "density": 0.0,
            "n_components": 0,
            "largest_cc_size": 0,
            "largest_cc_frac": 0.0,
            "mean_degree_induced": 0.0,
        }

    subA = A_csr[selected_idx][:, selected_idx].tocsr()

    # 无向图边数 = nnz / 2
    n_edges = int(subA.nnz // 2)

    if n_selected <= 1:
        density = 0.0
    else:
        density = float(2.0 * n_edges / (n_selected * (n_selected - 1)))

    degrees = np.asarray(subA.sum(axis=1)).ravel()
    mean_degree = float(degrees.mean()) if n_selected > 0 else 0.0

    n_comp, labels = connected_components(subA, directed=False, return_labels=True)
    if n_selected > 0:
        comp_sizes = np.bincount(labels, minlength=n_comp)
        largest_cc_size = int(comp_sizes.max()) if comp_sizes.size > 0 else 0
        largest_cc_frac = float(largest_cc_size / n_selected)
    else:
        largest_cc_size = 0
        largest_cc_frac = 0.0

    return {
        "n_selected": n_selected,
        "n_edges_induced": n_edges,
        "density": density,
        "n_components": int(n_comp),
        "largest_cc_size": largest_cc_size,
        "largest_cc_frac": largest_cc_frac,
        "mean_degree_induced": mean_degree,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_files = sorted(run_dir.glob("selected_*.npy"))
    if not selected_files:
        raise FileNotFoundError(f"在 {run_dir} 下没有找到 selected_*.npy")

    A_csr, graph_source = load_adjacency(data_dir)
    n_features = A_csr.shape[0]

    rows = []
    for path in selected_files:
        arr = np.load(path, allow_pickle=True)
        idx = parse_selected(arr, n_features)
        metrics = compute_metrics(A_csr, idx)

        row = {
            "selected_file": str(path),
            "seed_hint": path.stem,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    per_seed_path = out_dir / "per_seed_module_metrics.tsv"
    df.to_csv(per_seed_path, sep="\t", index=False)

    summary_rows = []
    metric_cols = [
        "n_selected",
        "n_edges_induced",
        "density",
        "n_components",
        "largest_cc_size",
        "largest_cc_frac",
        "mean_degree_induced",
    ]
    for col in metric_cols:
        vals = df[col].astype(float).values
        summary_rows.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_module_metrics.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    run_info = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "graph_source": graph_source,
        "n_selected_files": len(selected_files),
        "n_features": int(n_features),
        "per_seed_file": str(per_seed_path),
        "summary_file": str(summary_path),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print("Graph source:", graph_source)
    print("n_selected_files =", len(selected_files))
    print("Saved:", per_seed_path)
    print("Saved:", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()