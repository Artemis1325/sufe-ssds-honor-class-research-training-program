from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RUNS = {
    "real": {
        "multiseed_json": RESULTS / "brca_real_multiseed20" / "multiseed_20260321-152109.json",
        "compare_dir": None,
    },
    "drop10": {
        "multiseed_json": RESULTS / "brca_drop10_s42_multiseed20" / "multiseed_20260321-153310.json",
        "compare_dir": RESULTS / "gene_compare_brca_real_multiseed20_vs_brca_drop10_s42_multiseed20",
    },
    "drop30": {
        "multiseed_json": RESULTS / "brca_drop30_s42_multiseed20" / "multiseed_20260321-153922.json",
        "compare_dir": RESULTS / "gene_compare_brca_real_multiseed20_vs_brca_drop30_s42_multiseed20",
    },
    "drop50": {
        "multiseed_json": RESULTS / "brca_drop50_s42_multiseed20" / "multiseed_20260321-154547.json",
        "compare_dir": RESULTS / "gene_compare_brca_real_multiseed20_vs_brca_drop50_s42_multiseed20",
    },
}


def load_json(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def find_final_epoch_metrics(obj: dict):
    """
    兼容两种情况：
    1) per_seed 里每个 seed 直接有 final
    2) per_seed 里有 history / epochs，需要取最后一个epoch
    """
    per_seed = obj.get("per_seed", [])
    rows = []

    for item in per_seed:
        seed = item.get("seed", None)

        if isinstance(item.get("final"), dict):
            final = item["final"]
        elif isinstance(item.get("history"), list) and len(item["history"]) > 0:
            final = item["history"][-1]
        elif isinstance(item.get("epochs"), list) and len(item["epochs"]) > 0:
            final = item["epochs"][-1]
        else:
            raise ValueError(f"Cannot find final metrics for seed item: keys={list(item.keys())}")

        acc = final.get("acc") or final.get("ACC")
        auc = final.get("auc") or final.get("AUC")
        nnz = (
            final.get("nnz_theta")
            or final.get("nnz")
            or final.get("NNZ")
        )
        le = (
            final.get("lap_energy")
            or final.get("laplacian_energy")
            or final.get("LE")
        )

        rows.append({
            "seed": seed,
            "acc": float(acc) if acc is not None else None,
            "auc": float(auc) if auc is not None else None,
            "nnz": float(nnz) if nnz is not None else None,
            "lap_energy": float(le) if le is not None else None,
        })

    df = pd.DataFrame(rows)
    return df


def read_list_count(fp: Path):
    if not fp.exists():
        return None
    with open(fp, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]
    return len(lines)


def module_density(compare_dir: Path, which: str):
    if compare_dir is None:
        return None

    if which == "real":
        fp = compare_dir / "real_biased_top200_from_all.tsv"
    else:
        fp = compare_dir / "drop_biased_top200_from_all.tsv"

    if not fp.exists():
        return None

    df = pd.read_csv(fp, sep="\t")
    n = len(df)
    # 这里只统计 gene 数量；真正 density 用你之前 local_modules 输出手填更稳
    return n


def main():
    summary_rows = []

    for name, meta in RUNS.items():
        obj = load_json(meta["multiseed_json"])
        df = find_final_epoch_metrics(obj)

        row = {
            "graph_prior": name,
            "n_seeds": len(df),
            "acc_mean": df["acc"].mean(),
            "acc_std": df["acc"].std(ddof=1),
            "auc_mean": df["auc"].mean(),
            "auc_std": df["auc"].std(ddof=1),
            "nnz_mean": df["nnz"].mean(),
            "nnz_std": df["nnz"].std(ddof=1),
            "lap_energy_mean": df["lap_energy"].mean(),
            "lap_energy_std": df["lap_energy"].std(ddof=1),
            "stable_set_size_freq_ge_5": None,
            "shared_with_real_freq_ge_5": None,
        }

        compare_dir = meta["compare_dir"]
        if compare_dir is not None:
            row["stable_set_size_freq_ge_5"] = read_list_count(compare_dir / "drop_only_minfreq5.txt")
            # 注意：这里只是 drop_only，不是 stable set size；下面会再补正
            shared = read_list_count(compare_dir / "shared_minfreq5.txt")
            drop_only = read_list_count(compare_dir / "drop_only_minfreq5.txt")
            if shared is not None and drop_only is not None:
                row["stable_set_size_freq_ge_5"] = shared + drop_only
                row["shared_with_real_freq_ge_5"] = shared
        else:
            # real 的 stable set size 直接用和 drop10 对比目录里的 real_only + shared 反推
            cdir = RUNS["drop10"]["compare_dir"]
            shared = read_list_count(cdir / "shared_minfreq5.txt")
            real_only = read_list_count(cdir / "real_only_minfreq5.txt")
            if shared is not None and real_only is not None:
                row["stable_set_size_freq_ge_5"] = shared + real_only
                row["shared_with_real_freq_ge_5"] = shared + real_only  # 对 real 自己来说就是全体

        summary_rows.append(row)

    out_df = pd.DataFrame(summary_rows)

    out_fp = RESULTS / "brca_drop_robustness_summary.tsv"
    out_df.to_csv(out_fp, sep="\t", index=False)

    print("Saved summary to:", out_fp)
    print()
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nManual densities to fill from local module output:")
    print("real vs drop10: real=0.004322, drop=0.002701")
    print("real vs drop30: real=0.003518, drop=0.002714")
    print("real vs drop50: real=0.002663, drop=0.002764")


if __name__ == "__main__":
    main()