import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = list(range(42, 62))

GROUPS = {
    "Reactome graph": [f"brca_real_s{s}_g1e3" for s in SEEDS],
    "Random graph": [f"brca_random_s{s}_g1e3" for s in SEEDS],
    "Degree-preserving random graph": [f"brca_degnull_s{s}_g1e3" for s in SEEDS],
}

METRICS = ["acc", "auc", "nnz_theta", "lap_energy"]

def load_final(run_name: str):
    run_dir = RESULTS / run_name
    files = sorted(run_dir.glob("run_*.json"))
    if not files:
        raise FileNotFoundError(f"No result json found in {run_dir}")
    with open(files[-1], "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["final"]

def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"

def main():
    summary = {}

    for group_name, run_names in GROUPS.items():
        rows = [load_final(rn) for rn in run_names]
        summary[group_name] = {}
        for k in METRICS:
            vals = np.array([r[k] for r in rows], dtype=float)
            summary[group_name][k] = {
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "values": vals.tolist(),
            }

    print("\n=== Detailed summary ===")
    for group_name in GROUPS:
        print(f"\n[{group_name}]")
        for k in METRICS:
            item = summary[group_name][k]
            print(f"{k}: mean={item['mean']:.6f}, std={item['std']:.6f}")

    print("\n=== LaTeX-ready rows ===")
    for group_name in GROUPS:
        acc = summary[group_name]["acc"]
        auc = summary[group_name]["auc"]
        nnz = summary[group_name]["nnz_theta"]
        le = summary[group_name]["lap_energy"]
        print(
            f"{group_name} & "
            f"${acc['mean']:.4f} \\pm {acc['std']:.4f}$ & "
            f"${auc['mean']:.4f} \\pm {auc['std']:.4f}$ & "
            f"${nnz['mean']:.1f} \\pm {nnz['std']:.1f}$ & "
            f"${le['mean']:.1f} \\pm {le['std']:.1f}$ \\\\"
        )

if __name__ == "__main__":
    main()