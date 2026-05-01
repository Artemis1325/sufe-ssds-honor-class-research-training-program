import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_file", type=str, required=True)
    parser.add_argument("--comp_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--real_tag", type=str, default="real")
    parser.add_argument("--comp_tag", type=str, default="comp")
    args = parser.parse_args()

    real_df = pd.read_csv(args.real_file, sep="\t")
    comp_df = pd.read_csv(args.comp_file, sep="\t")

    metrics = [
        "lap_energy_norm",
        "mean_abs_diff_on_edges",
        "mean_sq_diff_on_edges",
    ]

    rows = []
    for m in metrics:
        x = real_df[m].astype(float).values
        y = comp_df[m].astype(float).values

        t_res = ttest_ind(x, y, equal_var=False)
        mw_res = mannwhitneyu(x, y, alternative="two-sided")

        rows.append({
            "metric": m,
            f"{args.real_tag}_mean": x.mean(),
            f"{args.comp_tag}_mean": y.mean(),
            "mean_ratio_comp_over_real": y.mean() / x.mean() if x.mean() != 0 else float("inf"),
            "welch_t_pvalue": t_res.pvalue,
            "mannwhitney_pvalue": mw_res.pvalue,
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)

    print("Saved:", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()