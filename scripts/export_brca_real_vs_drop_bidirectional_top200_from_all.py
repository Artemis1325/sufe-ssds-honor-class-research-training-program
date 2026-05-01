from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

COMPARE_DIRS = [
    RESULTS_DIR / "gene_compare_brca_real_multiseed20_vs_brca_drop10_s42_multiseed20",
    RESULTS_DIR / "gene_compare_brca_real_multiseed20_vs_brca_drop30_s42_multiseed20",
    RESULTS_DIR / "gene_compare_brca_real_multiseed20_vs_brca_drop50_s42_multiseed20",
]

TOPN = 200

for compare_dir in COMPARE_DIRS:
    in_fp = compare_dir / "real_vs_drop_diff_all.tsv"
    if not in_fp.exists():
        raise FileNotFoundError(f"Missing file: {in_fp}")

    df = pd.read_csv(in_fp, sep="\t")

    real_df = (
        df[df["diff"] > 0]
        .sort_values(["diff", "real_freq", "drop_freq", "gene"], ascending=[False, False, True, True])
        .head(TOPN)
        .copy()
    )
    drop_df = (
        df[df["diff"] < 0]
        .sort_values(["diff", "drop_freq", "real_freq", "gene"], ascending=[True, False, True, True])
        .head(TOPN)
        .copy()
    )

    real_out = compare_dir / "real_biased_top200_from_all.tsv"
    drop_out = compare_dir / "drop_biased_top200_from_all.tsv"

    real_df.to_csv(real_out, sep="\t", index=False)
    drop_df.to_csv(drop_out, sep="\t", index=False)

    print(f"[{compare_dir.name}]")
    print(f"real_count = {len(real_df)}")
    print(f"drop_count = {len(drop_df)}")
    print(f"saved: {real_out}")
    print(f"saved: {drop_out}")
    print("-" * 80)