import os
from pathlib import Path

import pandas as pd
import gseapy as gp


ROOT = Path(__file__).resolve().parents[1]
CMP_DIR = ROOT / "results" / "gene_compare_brca_real_multiseed20_vs_brca_drop30_s42_multiseed20"
IN_DIR = CMP_DIR / "enrichment_inputs"
OUT_DIR = CMP_DIR / "enrichment_results"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(gene_file: Path, tag: str, gene_sets: dict):
    genes = [x.strip() for x in gene_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"[run] {tag}: {len(genes)} genes")

    all_tables = []
    for gs_name, gs_lib in gene_sets.items():
        out_subdir = OUT_DIR / f"{tag}__{gs_name}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gs_lib,
            organism="human",
            outdir=str(out_subdir),
            cutoff=0.5,
        )

        if enr is None or enr.results is None or enr.results.empty:
            print(f"  [warn] no results for {tag} / {gs_name}")
            continue

        df = enr.results.copy()
        df["input_set"] = tag
        df["library"] = gs_name
        all_tables.append(df)

        keep_cols = [c for c in [
            "Term", "Adjusted P-value", "P-value", "Odds Ratio",
            "Combined Score", "Genes", "Overlap"
        ] if c in df.columns]
        preview = df.loc[:, keep_cols].head(10)
        print(f"\n[top terms] {tag} / {gs_name}")
        print(preview.to_string(index=False))

    if all_tables:
        merged = pd.concat(all_tables, axis=0, ignore_index=True)
        save_fp = OUT_DIR / f"{tag}_enrichment_all.tsv"
        merged.to_csv(save_fp, sep="\t", index=False)
        print(f"\n[saved] {save_fp}")
        return merged
    return None


def make_summary(dfs: list):
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        return

    big = pd.concat(frames, axis=0, ignore_index=True)

    cand_cols = [c for c in ["Adjusted P-value", "P-value", "Odds Ratio", "Combined Score"] if c in big.columns]
    for c in cand_cols:
        big[c] = pd.to_numeric(big[c], errors="coerce")

    sort_cols = [c for c in ["input_set", "library", "Adjusted P-value", "P-value", "Combined Score"] if c in big.columns]
    ascending = []
    for c in sort_cols:
        if c in ["Adjusted P-value", "P-value"]:
            ascending.append(True)
        elif c == "Combined Score":
            ascending.append(False)
        else:
            ascending.append(True)

    if sort_cols:
        big = big.sort_values(sort_cols, ascending=ascending)

    summary_fp = OUT_DIR / "enrichment_run_summary.tsv"
    big.to_csv(summary_fp, sep="\t", index=False)
    print(f"\n[saved summary] {summary_fp}")

    for tag in ["real_biased_top200", "drop_biased_top200"]:
        sub = big[big["input_set"] == tag].copy()
        if sub.empty:
            continue

        cols = [c for c in [
            "library", "Term", "Adjusted P-value", "P-value",
            "Odds Ratio", "Combined Score", "Overlap", "Genes"
        ] if c in sub.columns]
        print(f"\n{'='*90}")
        print(f"[best terms for {tag}]")
        print(sub.loc[:, cols].head(12).to_string(index=False))


def main():
    gene_sets = {
        "Reactome_2022": "Reactome_2022",
        "GO_Biological_Process_2023": "GO_Biological_Process_2023",
    }

    real_df = run_one(
        IN_DIR / "real_biased_top200.txt",
        "real_biased_top200",
        gene_sets,
    )
    drop_df = run_one(
        IN_DIR / "drop_biased_top200.txt",
        "drop_biased_top200",
        gene_sets,
    )

    make_summary([real_df, drop_df])


if __name__ == "__main__":
    main()