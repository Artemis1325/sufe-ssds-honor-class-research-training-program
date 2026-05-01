import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind, mannwhitneyu


def read_feature_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_gene_list_from_tsv(path):
    df = pd.read_csv(path, sep="\t")
    candidate_cols = [c for c in df.columns if c.lower() in {"gene", "gene_symbol", "symbol"}]
    col = candidate_cols[0] if candidate_cols else df.columns[0]
    genes = df[col].astype(str).str.strip()
    genes = genes[genes != ""].tolist()

    out = []
    seen = set()
    for g in genes:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


def zscore_by_gene(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def safe_auc(y_true, score):
    auc = roc_auc_score(y_true, score)
    direction = "tumor_higher"
    if auc < 0.5:
        auc = 1.0 - auc
        direction = "normal_higher"
    return auc, direction


def compute_module_stats(X_all, y, feature_symbols, gene_set_path):
    symbol_to_idx = {g: i for i, g in enumerate(feature_symbols)}
    genes = read_gene_list_from_tsv(gene_set_path)
    present = [g for g in genes if g in symbol_to_idx]
    idx = [symbol_to_idx[g] for g in present]

    X_sub = X_all[:, idx]
    Xz = zscore_by_gene(X_sub)
    score = Xz.mean(axis=1)

    tumor_score = score[y == 1]
    normal_score = score[y == 0]

    auc, direction = safe_auc(y, score)
    t_p = ttest_ind(tumor_score, normal_score, equal_var=False).pvalue
    mw_p = mannwhitneyu(tumor_score, normal_score, alternative="two-sided").pvalue

    return {
        "genes": genes,
        "present": present,
        "Xz": Xz,
        "score": score,
        "auc_abs_direction": auc,
        "direction": direction,
        "tumor_mean_score": float(np.mean(tumor_score)),
        "normal_mean_score": float(np.mean(normal_score)),
        "score_gap_tumor_minus_normal": float(np.mean(tumor_score) - np.mean(normal_score)),
        "welch_t_pvalue": float(t_p),
        "mannwhitney_pvalue": float(mw_p),
    }


def plot_pca(Xz, y, out_png, title):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xz)

    plt.figure(figsize=(6, 5))
    normal_mask = (y == 0)
    tumor_mask = (y == 1)
    plt.scatter(pcs[normal_mask, 0], pcs[normal_mask, 1], label="normal")
    plt.scatter(pcs[tumor_mask, 0], pcs[tumor_mask, 1], label="tumor")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def plot_heatmap(Xz, y, genes, out_png, out_gene_tsv, top_var_genes=80, title=""):
    sample_order = np.argsort(y)
    Xz_ord = Xz[sample_order]
    y_ord = y[sample_order]

    gene_var = Xz_ord.var(axis=0)
    gene_order = np.argsort(-gene_var)
    Xz_ord_gene = Xz_ord[:, gene_order]
    genes_ord = [genes[i] for i in gene_order]

    topk = min(top_var_genes, Xz_ord_gene.shape[1])
    Xz_heat = Xz_ord_gene[:, :topk]
    genes_heat = genes_ord[:topk]

    pd.DataFrame({"gene": genes_heat}).to_csv(out_gene_tsv, sep="\t", index=False)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(Xz_heat.T, aspect="auto")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.yticks(np.arange(len(genes_heat)), genes_heat, fontsize=7)
    plt.xticks([])
    plt.xlabel("Samples (ordered: normal -> tumor)")
    plt.ylabel("Genes")
    plt.title(title)

    n_normal = int((y_ord == 0).sum())
    plt.axvline(x=n_normal - 0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_dir", type=str, required=True)
    parser.add_argument("--real_gene_set", type=str, required=True)
    parser.add_argument("--drop_gene_set", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--top_var_genes", type=int, default=80)
    args = parser.parse_args()

    external_dir = Path(args.external_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(external_dir / "X_graph.npy")
    y = np.load(external_dir / "y.npy").astype(int)
    feature_symbols = read_feature_symbols(external_dir / "feature_graph_symbols.txt")

    real_res = compute_module_stats(X, y, feature_symbols, args.real_gene_set)
    drop_res = compute_module_stats(X, y, feature_symbols, args.drop_gene_set)

    pd.DataFrame({"gene": real_res["present"]}).to_csv(out_dir / "real_present_genes.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": drop_res["present"]}).to_csv(out_dir / "drop_present_genes.tsv", sep="\t", index=False)

    pc1_r, pc2_r = plot_pca(
        real_res["Xz"], y,
        out_dir / "real_pca.png",
        "GSE15852 PCA on real-biased transferred genes"
    )
    pc1_d, pc2_d = plot_pca(
        drop_res["Xz"], y,
        out_dir / "drop_pca.png",
        "GSE15852 PCA on drop-biased transferred genes"
    )

    plot_heatmap(
        real_res["Xz"], y, real_res["present"],
        out_dir / "real_heatmap.png",
        out_dir / "real_heatmap_top_variable_genes.tsv",
        top_var_genes=args.top_var_genes,
        title=f"GSE15852 heatmap (real-biased, top {args.top_var_genes} variable genes)"
    )
    plot_heatmap(
        drop_res["Xz"], y, drop_res["present"],
        out_dir / "drop_heatmap.png",
        out_dir / "drop_heatmap_top_variable_genes.tsv",
        top_var_genes=args.top_var_genes,
        title=f"GSE15852 heatmap (drop-biased, top {args.top_var_genes} variable genes)"
    )

    scores_long = pd.DataFrame({
        "sample_index": list(range(len(y))) * 2,
        "y": list(y) * 2,
        "label": list(np.where(y == 1, "tumor", "normal")) * 2,
        "module_score": list(real_res["score"]) + list(drop_res["score"]),
        "set_name": ["real_biased"] * len(y) + ["drop_biased"] * len(y),
    })
    scores_long.to_csv(out_dir / "module_scores_long.tsv", sep="\t", index=False)

    summary = pd.DataFrame([
        {
            "set_name": "real_biased",
            "n_genes_total": len(real_res["genes"]),
            "n_present": len(real_res["present"]),
            "coverage": len(real_res["present"]) / max(len(real_res["genes"]), 1),
            "tumor_mean_score": real_res["tumor_mean_score"],
            "normal_mean_score": real_res["normal_mean_score"],
            "score_gap_tumor_minus_normal": real_res["score_gap_tumor_minus_normal"],
            "auc_abs_direction": real_res["auc_abs_direction"],
            "direction": real_res["direction"],
            "welch_t_pvalue": real_res["welch_t_pvalue"],
            "mannwhitney_pvalue": real_res["mannwhitney_pvalue"],
            "pc1_explained_variance_ratio": pc1_r,
            "pc2_explained_variance_ratio": pc2_r,
        },
        {
            "set_name": "drop_biased",
            "n_genes_total": len(drop_res["genes"]),
            "n_present": len(drop_res["present"]),
            "coverage": len(drop_res["present"]) / max(len(drop_res["genes"]), 1),
            "tumor_mean_score": drop_res["tumor_mean_score"],
            "normal_mean_score": drop_res["normal_mean_score"],
            "score_gap_tumor_minus_normal": drop_res["score_gap_tumor_minus_normal"],
            "auc_abs_direction": drop_res["auc_abs_direction"],
            "direction": drop_res["direction"],
            "welch_t_pvalue": drop_res["welch_t_pvalue"],
            "mannwhitney_pvalue": drop_res["mannwhitney_pvalue"],
            "pc1_explained_variance_ratio": pc1_d,
            "pc2_explained_variance_ratio": pc2_d,
        },
    ])
    summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False)

    print("Saved:", out_dir / "summary.tsv")
    print("Saved:", out_dir / "module_scores_long.tsv")
    print("Saved:", out_dir / "real_pca.png")
    print("Saved:", out_dir / "drop_pca.png")
    print("Saved:", out_dir / "real_heatmap.png")
    print("Saved:", out_dir / "drop_heatmap.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()