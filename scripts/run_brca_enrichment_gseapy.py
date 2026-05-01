import os
import sys
import traceback
from pathlib import Path

import pandas as pd

try:
    import gseapy as gp
except ImportError:
    print("ERROR: gseapy is not installed.")
    print("Please install it first, e.g. pip install gseapy")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "brca_gene_compare_g1e3" / "enrichment_inputs"
OUT_DIR = ROOT / "results" / "brca_gene_compare_g1e3" / "enrichment_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_LISTS = {
    "real_biased_top200": INPUT_DIR / "real_biased_top200.txt",
    "degnull_biased_top200": INPUT_DIR / "degnull_biased_top200.txt",
}

LIBRARIES = [
    "Reactome_2022",
    "GO_Biological_Process_2023",
]

def read_gene_list(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        genes = [x.strip().upper() for x in f if x.strip()]
    seen = set()
    uniq = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq

def pick_qcol(df: pd.DataFrame):
    candidates = [
        "Adjusted P-value",
        "Adjusted P-value ",
        "P-value",
        "Old P-value",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower().strip()
        if "adjusted" in lc and "p" in lc:
            return c
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"p-value", "pvalue", "p value"}:
            return c
    return None

def pick_termcol(df: pd.DataFrame):
    for c in ["Term", "term"]:
        if c in df.columns:
            return c
    return df.columns[0]

def pick_overlapcol(df: pd.DataFrame):
    for c in ["Overlap", "overlap"]:
        if c in df.columns:
            return c
    return None

def summarize_result(label: str, lib: str, df: pd.DataFrame):
    print("-" * 100)
    print(f"[{label}] [{lib}]")
    print(f"rows: {len(df)}")
    print("columns:", list(df.columns))

    qcol = pick_qcol(df)
    termcol = pick_termcol(df)
    overlapcol = pick_overlapcol(df)

    if qcol is not None:
        try:
            df = df.sort_values(qcol, ascending=True).reset_index(drop=True)
        except Exception:
            pass

    topn = min(10, len(df))
    if topn == 0:
        print("No enrichment terms returned.")
        return

    print(f"top {topn} terms:")
    for i in range(topn):
        row = df.iloc[i]
        term = row[termcol]
        qv = row[qcol] if qcol is not None else "NA"
        ov = row[overlapcol] if overlapcol is not None else "NA"
        print(f"{i+1:02d}. {term} | q={qv} | overlap={ov}")

def main():
    summary_rows = []

    for label, fp in GENE_LISTS.items():
        genes = read_gene_list(fp)
        print("=" * 100)
        print(f"[gene list] {label}")
        print(f"file: {fp}")
        print(f"n_genes: {len(genes)}")
        print(f"preview: {genes[:10]}")

        for lib in LIBRARIES:
            subdir = OUT_DIR / f"{label}__{lib}"
            subdir.mkdir(parents=True, exist_ok=True)

            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=lib,
                    organism="human",
                    outdir=str(subdir),
                    no_plot=True,
                )
                res = enr.results.copy()
                out_tsv = subdir / "enrichment_results.tsv"
                res.to_csv(out_tsv, sep="\t", index=False)

                summarize_result(label, lib, res)

                qcol = pick_qcol(res)
                termcol = pick_termcol(res)

                if qcol is not None and len(res) > 0:
                    try:
                        res2 = res.sort_values(qcol, ascending=True).reset_index(drop=True)
                    except Exception:
                        res2 = res.reset_index(drop=True)
                    top_term = res2.iloc[0][termcol]
                    top_q = res2.iloc[0][qcol]
                    sig_n = int((pd.to_numeric(res[qcol], errors="coerce") < 0.05).sum())
                else:
                    top_term = "NA"
                    top_q = "NA"
                    sig_n = 0

                summary_rows.append({
                    "gene_list": label,
                    "library": lib,
                    "n_input_genes": len(genes),
                    "n_terms_returned": len(res),
                    "n_sig_fdr_lt_0_05": sig_n,
                    "top_term": top_term,
                    "top_q_or_p": top_q,
                    "result_file": str(out_tsv),
                })

            except Exception as e:
                print("-" * 100)
                print(f"[ERROR] {label} | {lib}")
                print(repr(e))
                traceback.print_exc()

                summary_rows.append({
                    "gene_list": label,
                    "library": lib,
                    "n_input_genes": len(genes),
                    "n_terms_returned": -1,
                    "n_sig_fdr_lt_0_05": -1,
                    "top_term": f"ERROR: {repr(e)}",
                    "top_q_or_p": "NA",
                    "result_file": "NA",
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_fp = OUT_DIR / "enrichment_run_summary.tsv"
    summary_df.to_csv(summary_fp, sep="\t", index=False)

    print("=" * 100)
    print("[saved summary]")
    print(summary_fp)
    print(summary_df)

if __name__ == "__main__":
    main()