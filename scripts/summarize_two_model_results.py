from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lasso-dir", required=True)
    p.add_argument("--graph-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-label", required=True)
    p.add_argument("--prefix", required=True)
    return p.parse_args()


def load_multiseed_metrics(run_dir: Path, model_name: str) -> dict:
    json_fp = next(run_dir.glob("multiseed_*.json"))
    data = json.loads(json_fp.read_text(encoding="utf-8"))
    finals = [row["final"] for row in data["per_seed"]]
    frame = pd.DataFrame(finals)
    frame = frame.rename(columns={"lap_energy": "laplacian_energy"})
    return {
        "model": model_name,
        "run_dir": str(run_dir),
        "n_seeds": int(len(frame)),
        "acc_mean": float(frame["acc"].mean()),
        "acc_std": float(frame["acc"].std(ddof=1)),
        "auc_mean": float(frame["auc"].mean()),
        "auc_std": float(frame["auc"].std(ddof=1)),
        "nnz_mean": float(frame["nnz_theta"].mean()),
        "nnz_std": float(frame["nnz_theta"].std(ddof=1)),
        "lap_mean": float(frame["laplacian_energy"].mean()),
        "lap_std": float(frame["laplacian_energy"].std(ddof=1)),
        "time_mean": float(frame["elapsed_seconds"].mean()),
        "time_std": float(frame["elapsed_seconds"].std(ddof=1)),
        "per_seed": frame.to_dict(orient="records"),
    }


def plot_metric(summary: pd.DataFrame, metric: str, out_fp: Path, ylabel: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    means = summary[f"{metric}_mean"].tolist()
    stds = summary[f"{metric}_std"].tolist()
    labels = summary["model"].tolist()
    colors = ["#9CA3AF", "#2563EB"]

    ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="#374151", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        load_multiseed_metrics(Path(args.lasso_dir), "LassoNet"),
        load_multiseed_metrics(Path(args.graph_dir), "Graph-LassoNet"),
    ]
    detail_rows = []
    for row in rows:
        for i, seed_row in enumerate(row["per_seed"]):
            detail_rows.append(
                {
                    "model": row["model"],
                    "seed_idx": i,
                    "acc": seed_row["acc"],
                    "auc": seed_row["auc"],
                    "nnz": seed_row["nnz_theta"],
                    "laplacian_energy": seed_row["laplacian_energy"],
                    "elapsed_seconds": seed_row["elapsed_seconds"],
                }
            )

    summary = pd.DataFrame([{k: v for k, v in row.items() if k != "per_seed"} for row in rows])
    details = pd.DataFrame(detail_rows)

    summary.to_csv(out_dir / "summary.csv", index=False)
    details.to_csv(out_dir / "per_seed.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_metric(summary, "acc", out_dir / f"{args.prefix}_acc_compare.png", f"{args.dataset_label} ACC")
    plot_metric(summary, "auc", out_dir / f"{args.prefix}_auc_compare.png", f"{args.dataset_label} AUC")
    plot_metric(summary, "nnz", out_dir / f"{args.prefix}_nnz_compare.png", f"{args.dataset_label} NNZ")
    plot_metric(summary, "lap", out_dir / f"{args.prefix}_lap_compare.png", f"{args.dataset_label} Laplacian energy")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
