import json
from pathlib import Path
from math import sqrt

from scipy.stats import ttest_rel, wilcoxon


JSON_PATH = r"D:\LapReg_lassonet_project\results\brca_ablation_tcga_20seed\ablation_20260323-220645.json"


def cohens_d_paired(x, y):
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n < 2:
        return 0.0
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    sd = sqrt(var) if var > 0 else 0.0
    return mean_diff / sd if sd > 0 else 0.0


def main():
    data = json.loads(Path(JSON_PATH).read_text(encoding="utf-8"))
    summary = data["summary"]

    lap = {
        k: summary[k]["metrics"]["lap_energy"]["values"]
        for k in ["mlp", "graph_mlp", "lassonet", "graph_lassonet"]
    }

    pairs = [
        ("graph_mlp", "mlp"),
        ("lassonet", "mlp"),
        ("graph_lassonet", "mlp"),
        ("graph_lassonet", "graph_mlp"),
        ("graph_lassonet", "lassonet"),
    ]

    for a, b in pairs:
        x = lap[a]
        y = lap[b]

        t_stat, t_p = ttest_rel(x, y)
        try:
            w_stat, w_p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        d = cohens_d_paired(x, y)
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        rel = (mean_x - mean_y) / mean_y * 100.0

        print(f"{a} vs {b}")
        print(f"  mean: {mean_x:.4f} vs {mean_y:.4f}")
        print(f"  relative change: {rel:.2f}%")
        print(f"  paired t-test p = {t_p:.6g}")
        print(f"  wilcoxon p      = {w_p:.6g}")
        print(f"  paired Cohen's d = {d:.4f}")
        print()


if __name__ == "__main__":
    main()