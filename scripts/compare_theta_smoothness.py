import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_graph(data_dir: Path):
    a_path = data_dir / "A.npz"
    l_path = data_dir / "L.npz"

    if not a_path.exists():
        raise FileNotFoundError(f"找不到 {a_path}")
    if not l_path.exists():
        raise FileNotFoundError(f"找不到 {l_path}")

    A = sparse.load_npz(a_path).tocsr().astype(np.float64)
    L = sparse.load_npz(l_path).tocsr().astype(np.float64)

    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()

    # 二值化邻接，只保留有没有边
    A.data[:] = 1.0
    return A, L


def compute_theta_metrics(theta: np.ndarray, A: sparse.csr_matrix, L: sparse.csr_matrix, eps=1e-12):
    theta = np.asarray(theta).reshape(-1).astype(np.float64)

    l2_norm_sq = float(np.dot(theta, theta))
    lap_energy = float(theta @ (L @ theta))
    lap_energy_norm = float(lap_energy / max(l2_norm_sq, eps))

    n_nonzero_theta = int(np.count_nonzero(np.abs(theta) > 1e-12))

    # 只遍历上三角边，避免重复
    A_coo = sparse.triu(A, k=1).tocoo()
    row = A_coo.row
    col = A_coo.col

    if len(row) == 0:
        mean_abs_diff = 0.0
        mean_sq_diff = 0.0
        n_edges = 0
    else:
        diff = theta[row] - theta[col]
        mean_abs_diff = float(np.mean(np.abs(diff)))
        mean_sq_diff = float(np.mean(diff ** 2))
        n_edges = int(len(diff))

    return {
        "n_nonzero_theta": n_nonzero_theta,
        "l2_norm_sq": l2_norm_sq,
        "lap_energy": lap_energy,
        "lap_energy_norm": lap_energy_norm,
        "mean_abs_diff_on_edges": mean_abs_diff,
        "mean_sq_diff_on_edges": mean_sq_diff,
        "n_graph_edges_used": n_edges,
    }


def summarize_df(df: pd.DataFrame, metric_cols):
    rows = []
    for col in metric_cols:
        vals = df[col].astype(float).values
        rows.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def process_run_dir(run_dir: Path, tag: str, A: sparse.csr_matrix, L: sparse.csr_matrix, out_dir: Path):
    theta_files = sorted(run_dir.glob("theta_*.npy"))
    if not theta_files:
        raise FileNotFoundError(f"在 {run_dir} 下没有找到 theta_*.npy")

    rows = []
    for path in theta_files:
        theta = np.load(path, allow_pickle=True)
        metrics = compute_theta_metrics(theta, A, L)
        row = {
            "group": tag,
            "theta_file": str(path),
            "seed_hint": path.stem,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [
        "n_nonzero_theta",
        "l2_norm_sq",
        "lap_energy",
        "lap_energy_norm",
        "mean_abs_diff_on_edges",
        "mean_sq_diff_on_edges",
        "n_graph_edges_used",
    ]
    summary = summarize_df(df, metric_cols)

    df.to_csv(out_dir / f"{tag}_per_seed.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / f"{tag}_summary.tsv", sep="\t", index=False)

    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_run_dir", type=str, required=True)
    parser.add_argument("--comp_run_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--real_tag", type=str, default="real")
    parser.add_argument("--comp_tag", type=str, default="comp")
    args = parser.parse_args()

    real_run_dir = Path(args.real_run_dir)
    comp_run_dir = Path(args.comp_run_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A, L = load_graph(data_dir)

    real_df, real_summary = process_run_dir(real_run_dir, args.real_tag, A, L, out_dir)
    comp_df, comp_summary = process_run_dir(comp_run_dir, args.comp_tag, A, L, out_dir)

    metric_cols = [
        "n_nonzero_theta",
        "l2_norm_sq",
        "lap_energy",
        "lap_energy_norm",
        "mean_abs_diff_on_edges",
        "mean_sq_diff_on_edges",
        "n_graph_edges_used",
    ]

    merged_rows = []
    real_map = {row["metric"]: row for _, row in real_summary.iterrows()}
    comp_map = {row["metric"]: row for _, row in comp_summary.iterrows()}
    for m in metric_cols:
        merged_rows.append({
            "metric": m,
            f"{args.real_tag}_mean": float(real_map[m]["mean"]),
            f"{args.real_tag}_std": float(real_map[m]["std"]),
            f"{args.comp_tag}_mean": float(comp_map[m]["mean"]),
            f"{args.comp_tag}_std": float(comp_map[m]["std"]),
        })

    compare_df = pd.DataFrame(merged_rows)
    compare_df.to_csv(out_dir / "compare_summary.tsv", sep="\t", index=False)

    run_info = {
        "real_run_dir": str(real_run_dir),
        "comp_run_dir": str(comp_run_dir),
        "data_dir": str(data_dir),
        "n_real_theta_files": len(real_df),
        "n_comp_theta_files": len(comp_df),
        "compare_summary_file": str(out_dir / "compare_summary.tsv"),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print("Saved:", out_dir / f"{args.real_tag}_per_seed.tsv")
    print("Saved:", out_dir / f"{args.real_tag}_summary.tsv")
    print("Saved:", out_dir / f"{args.comp_tag}_per_seed.tsv")
    print("Saved:", out_dir / f"{args.comp_tag}_summary.tsv")
    print("Saved:", out_dir / "compare_summary.tsv")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()