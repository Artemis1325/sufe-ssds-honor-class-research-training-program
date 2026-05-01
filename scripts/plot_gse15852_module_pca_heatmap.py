import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_feature_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_gene_list_from_tsv(path):
    df = pd.read_csv(path, sep="\t")
    candidate_cols = [c for c in df.columns if c.lower() in {"gene", "gene_symbol", "symbol"}]
    col = candidate_cols[0] if candidate_cols else df.columns[0]
    genes = df[col].astype(str).str.strip()
    genes = genes[genes != ""].tolist()
    # 去重但保序
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_dir", type=str, required=True)
    parser.add_argument("--gene_set", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--top_var_genes", type=int, default=80)
    args = parser.parse_args()

    external_dir = Path(args.external_dir)
    gene_set_path = Path(args.gene_set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(external_dir / "X_graph.npy")
    y = np.load(external_dir / "y.npy").astype(int)
    feature_symbols = read_feature_symbols(external_dir / "feature_graph_symbols.txt")

    symbol_to_idx = {g: i for i, g in enumerate(feature_symbols)}
    genes = read_gene_list_from_tsv(gene_set_path)
    present = [g for g in genes if g in symbol_to_idx]
    idx = [symbol_to_idx[g] for g in present]

    if len(idx) == 0:
        raise ValueError("没有可用基因落在 external feature space 中。")

    X_sub = X[:, idx]
    Xz = zscore_by_gene(X_sub)

    sample_order = np.argsort(y)  # normal(0) 在前, tumor(1) 在后
    Xz_ord = Xz[sample_order]
    y_ord = y[sample_order]

    # 再按基因方差排序，热图更容易看
    gene_var = Xz_ord.var(axis=0)
    gene_order = np.argsort(-gene_var)
    Xz_ord_gene = Xz_ord[:, gene_order]
    genes_ord = [present[i] for i in gene_order]

    topk = min(args.top_var_genes, Xz_ord_gene.shape[1])
    Xz_heat = Xz_ord_gene[:, :topk]
    genes_heat = genes_ord[:topk]

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xz)

    pca_df = pd.DataFrame({
        "sample_index": np.arange(len(y)),
        "y": y,
        "label": np.where(y == 1, "tumor", "normal"),
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
    })
    pca_df.to_csv(out_dir / "pca_scores.tsv", sep="\t", index=False)

    pd.DataFrame({"gene": present}).to_csv(out_dir / "present_genes.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": genes_heat}).to_csv(out_dir / "heatmap_top_variable_genes.tsv", sep="\t", index=False)

    # ---------- PCA plot ----------
    plt.figure(figsize=(6, 5))
    normal_mask = (y == 0)
    tumor_mask = (y == 1)
    plt.scatter(pcs[normal_mask, 0], pcs[normal_mask, 1], label="normal")
    plt.scatter(pcs[tumor_mask, 0], pcs[tumor_mask, 1], label="tumor")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("GSE15852 PCA on transferred module genes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_scatter.png", dpi=200)
    plt.close()

    # ---------- Heatmap ----------
    plt.figure(figsize=(12, 8))
    im = plt.imshow(Xz_heat.T, aspect="auto")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.yticks(np.arange(len(genes_heat)), genes_heat, fontsize=7)
    plt.xticks([])
    plt.xlabel("Samples (ordered: normal -> tumor)")
    plt.ylabel("Genes")
    plt.title(f"GSE15852 heatmap (top {topk} variable genes in transferred module)")

    # 标一条分界线
    n_normal = int((y_ord == 0).sum())
    plt.axvline(x=n_normal - 0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_top_variable_genes.png", dpi=200)
    plt.close()

    # 简单汇总
    summary = {
        "gene_set_path": str(gene_set_path),
        "n_samples": int(X.shape[0]),
        "n_present_genes": int(len(present)),
        "n_normal": int((y == 0).sum()),
        "n_tumor": int((y == 1).sum()),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "top_var_genes_for_heatmap": int(topk),
    }
    pd.Series(summary).to_csv(out_dir / "summary.tsv", sep="\t", header=False)

    print("Saved:", out_dir / "pca_scatter.png")
    print("Saved:", out_dir / "heatmap_top_variable_genes.png")
    print("Saved:", out_dir / "pca_scores.tsv")
    print("Saved:", out_dir / "heatmap_top_variable_genes.tsv")
    print("PC1 explained variance ratio =", pca.explained_variance_ratio_[0])
    print("PC2 explained variance ratio =", pca.explained_variance_ratio_[1])


if __name__ == "__main__":
    main()