import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "brca_graph_mode_compare_summary"
SEEDS = list(range(42, 62))

GROUPS = {
    "Fixed external graph": [f"brca_fixed_s{s}" for s in SEEDS],
    "Train-only Pearson graph": [f"brca_pearson_train_s{s}_k10" for s in SEEDS],
}

METRICS = ["acc", "auc", "nnz_theta", "lap_energy", "elapsed_seconds"]


def load_latest_run_json(run_name: str):
    run_dir = RESULTS / run_name
    files = sorted(run_dir.glob("run_*.json"))
    if not files:
        raise FileNotFoundError(f"No result json found in {run_dir}")
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows():
    rows = []
    for group_name, run_names in GROUPS.items():
        for seed, run_name in zip(SEEDS, run_names):
            obj = load_latest_run_json(run_name)
            row = {
                "group": group_name,
                "seed": seed,
                "run_name": run_name,
            }
            row.update(obj["final"])
            graph_info = obj.get("meta", {}).get("graph_info", {})
            row["graph_mode"] = graph_info.get("graph_mode", "")
            row["undirected_edges"] = graph_info.get("undirected_edges", np.nan)
            row["degree_mean"] = graph_info.get("degree_mean", np.nan)
            row["isolated_nodes"] = graph_info.get("isolated_nodes", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame):
    summary = df.groupby("group")[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return summary


def make_latex_rows(summary_df: pd.DataFrame):
    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['group']} & "
            f"${row['acc_mean']:.4f} \\pm {row['acc_std']:.4f}$ & "
            f"${row['auc_mean']:.4f} \\pm {row['auc_std']:.4f}$ & "
            f"${row['nnz_theta_mean']:.1f} \\pm {row['nnz_theta_std']:.1f}$ & "
            f"${row['lap_energy_mean']:.2f} \\pm {row['lap_energy_std']:.2f}$ \\\\"
        )
    return lines


def plot_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, filename: str):
    x = np.arange(len(summary_df))
    means = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
    stds = summary_df[f"{metric}_std"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=["#4C78A8", "#F58518"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["group"].tolist(), rotation=10, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"BRCA: {ylabel}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for rect, mean in zip(bars, means):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{mean:.4f}" if metric in {"acc", "auc"} else f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_rows()
    summary_df = build_summary(df)

    df.to_csv(OUT_DIR / "per_seed_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "groups": GROUPS,
        "summary": summary_df.to_dict(orient="records"),
        "latex_rows": make_latex_rows(summary_df),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plot_metric(summary_df, metric="acc", ylabel="Accuracy", filename="brca_acc_compare.png")
    plot_metric(summary_df, metric="auc", ylabel="AUC", filename="brca_auc_compare.png")
    plot_metric(summary_df, metric="nnz_theta", ylabel="Selected features", filename="brca_nnz_compare.png")

    print("\n=== BRCA graph mode summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== LaTeX rows ===")
    for line in payload["latex_rows"]:
        print(line)
    print(f"\nSaved summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
