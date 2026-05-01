import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind, mannwhitneyu


def read_feature_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        symbols = [line.strip() for line in f if line.strip()]
    return symbols


def read_gene_list_from_tsv(path):
    df = pd.read_csv(path, sep="\t")
    # 尽量自动识别 gene symbol 列
    candidate_cols = [c for c in df.columns if c.lower() in {"gene", "gene_symbol", "symbol"}]
    if candidate_cols:
        col = candidate_cols[0]
    else:
        col = df.columns[0]
    genes = df[col].astype(str).str.strip()
    genes = genes[genes != ""].tolist()
    return genes


def zscore_by_gene(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def safe_auc(y_true, score):
    # 默认 tumor=1, normal=0
    auc = roc_auc_score(y_true, score)
    # 若方向反了，翻到 >= 0.5，便于解释“可分离性”
    if auc < 0.5:
        auc = 1.0 - auc
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_dir", type=str, required=True)
    parser.add_argument("--gene_sets", type=str, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    external_dir = Path(args.external_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(external_dir / "X_graph.npy")
    y = np.load(external_dir / "y.npy").astype(int)
    feature_symbols = read_feature_symbols(external_dir / "feature_graph_symbols.txt")

    assert X.shape[1] == len(feature_symbols), "X_graph 列数与 feature_graph_symbols 不一致"
    assert set(np.unique(y)).issubset({0, 1}), "y 必须是 0/1 标签"

    symbol_to_idx = {g: i for i, g in enumerate(feature_symbols)}

    # 关键：只在 GSE 内部做按基因标准化，避免直接套 TCGA scaler
    Xz = zscore_by_gene(X)

    rows = []
    all_scores = []

    for gene_set_path in args.gene_sets:
        gene_set_path = Path(gene_set_path)
        set_name = gene_set_path.parent.name

        genes = read_gene_list_from_tsv(gene_set_path)
        genes_unique = []
        seen = set()
        for g in genes:
            if g not in seen:
                genes_unique.append(g)
                seen.add(g)

        present = [g for g in genes_unique if g in symbol_to_idx]
        missing = [g for g in genes_unique if g not in symbol_to_idx]

        if len(present) == 0:
            row = {
                "set_name": set_name,
                "gene_set_path": str(gene_set_path),
                "n_genes_total": len(genes_unique),
                "n_present": 0,
                "n_missing": len(missing),
                "coverage": 0.0,
                "tumor_mean_score": np.nan,
                "normal_mean_score": np.nan,
                "score_gap_tumor_minus_normal": np.nan,
                "auc_abs_direction": np.nan,
                "welch_t_pvalue": np.nan,
                "mannwhitney_pvalue": np.nan,
            }
            rows.append(row)
            continue

        idx = [symbol_to_idx[g] for g in present]
        score = Xz[:, idx].mean(axis=1)

        tumor_score = score[y == 1]
        normal_score = score[y == 0]

        auc = safe_auc(y, score)
        t_p = ttest_ind(tumor_score, normal_score, equal_var=False).pvalue
        mw_p = mannwhitneyu(tumor_score, normal_score, alternative="two-sided").pvalue

        row = {
            "set_name": set_name,
            "gene_set_path": str(gene_set_path),
            "n_genes_total": len(genes_unique),
            "n_present": len(present),
            "n_missing": len(missing),
            "coverage": len(present) / max(len(genes_unique), 1),
            "tumor_mean_score": float(np.mean(tumor_score)),
            "normal_mean_score": float(np.mean(normal_score)),
            "score_gap_tumor_minus_normal": float(np.mean(tumor_score) - np.mean(normal_score)),
            "auc_abs_direction": float(auc),
            "welch_t_pvalue": float(t_p),
            "mannwhitney_pvalue": float(mw_p),
        }
        rows.append(row)

        tmp = pd.DataFrame({
            "sample_index": np.arange(len(score)),
            "y": y,
            "label": np.where(y == 1, "tumor", "normal"),
            "module_score": score,
            "set_name": set_name,
        })
        all_scores.append(tmp)

        pd.DataFrame({"gene": present}).to_csv(
            out_dir / f"{set_name}__present_genes.tsv", sep="\t", index=False
        )
        pd.DataFrame({"gene": missing}).to_csv(
            out_dir / f"{set_name}__missing_genes.tsv", sep="\t", index=False
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["auc_abs_direction", "score_gap_tumor_minus_normal"],
        ascending=[False, False],
        na_position="last"
    )
    summary.to_csv(out_dir / "module_transfer_summary.tsv", sep="\t", index=False)

    if all_scores:
        pd.concat(all_scores, axis=0, ignore_index=True).to_csv(
            out_dir / "module_scores_long.tsv", sep="\t", index=False
        )

    payload = {
        "external_dir": str(external_dir),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "tumor_count": int((y == 1).sum()),
        "normal_count": int((y == 0).sum()),
        "n_gene_sets": len(args.gene_sets),
        "summary_file": str(out_dir / "module_transfer_summary.tsv"),
        "scores_file": str(out_dir / "module_scores_long.tsv"),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Saved:", out_dir / "module_transfer_summary.tsv")
    if all_scores:
        print("Saved:", out_dir / "module_scores_long.tsv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()