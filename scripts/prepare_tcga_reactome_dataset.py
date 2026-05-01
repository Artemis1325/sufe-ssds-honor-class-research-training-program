from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
ENSEMBL_MAP = ROOT / "datasets" / "tcga_brca" / "ensembl_to_symbol.tsv"
REACTOME_EDGE_FILE = ROOT / "datasets" / "tcga_brca" / "reactome_edges_symbol.tsv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--expr", required=True, help="Expression TSV/TSV.GZ from Xena/GDC.")
    p.add_argument("--phenotype", required=True, help="Phenotype TSV/TSV.GZ from Xena/GDC.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--sample-col", default="sample")
    p.add_argument("--sample-type-col", default="sample_type.samples")
    p.add_argument("--tumor-label", default="Primary Tumor")
    p.add_argument("--normal-label", default="Solid Tissue Normal")
    return p.parse_args()


def infer_sample_type(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) >= 4:
        sample_type_code = parts[3][:2]
        if sample_type_code == "01":
            return "Primary Tumor"
        if sample_type_code == "11":
            return "Solid Tissue Normal"
    return ""


def build_dataset(args):
    expr_path = Path(args.expr)
    phenotype_path = Path(args.phenotype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr = pd.read_csv(expr_path, sep="\t")
    gene_col = expr.columns[0]
    expr = expr.rename(columns={gene_col: "gene_id"})
    expr["gene_id"] = expr["gene_id"].astype(str).str.split(".").str[0]
    expr = expr.groupby("gene_id", as_index=False).mean(numeric_only=True)

    pheno = pd.read_csv(phenotype_path, sep="\t")
    sample_col = args.sample_col if args.sample_col in pheno.columns else pheno.columns[0]
    pheno = pheno.rename(columns={sample_col: "sample_id"})
    if args.sample_type_col in pheno.columns:
        pheno = pheno.rename(columns={args.sample_type_col: "sample_type"})
    else:
        pheno["sample_type"] = pheno["sample_id"].map(infer_sample_type)

    keep_types = {args.tumor_label, args.normal_label}
    pheno = pheno[pheno["sample_type"].isin(keep_types)].copy()

    expr_samples = [c for c in expr.columns[1:] if c in set(pheno["sample_id"])]
    pheno = pheno[pheno["sample_id"].isin(expr_samples)].copy()
    pheno["sample_id"] = pd.Categorical(pheno["sample_id"], categories=expr_samples, ordered=True)
    pheno = pheno.sort_values("sample_id").reset_index(drop=True)
    sample_ids = pheno["sample_id"].astype(str).tolist()

    ensembl_map = pd.read_csv(ENSEMBL_MAP, sep="\t")
    ensembl_map = ensembl_map[ensembl_map["gene_type"] == "protein_coding"].copy()
    ensembl_map["gene_symbol"] = ensembl_map["gene_symbol"].astype(str).str.upper().str.strip()
    ensembl_map = ensembl_map[ensembl_map["gene_symbol"] != ""]

    expr = expr.merge(ensembl_map[["ensembl_id", "gene_symbol"]], left_on="gene_id", right_on="ensembl_id", how="inner")
    expr = expr.drop(columns=["ensembl_id"])
    expr = expr.groupby("gene_symbol", as_index=False).mean(numeric_only=True)

    reactome_edges = pd.read_csv(REACTOME_EDGE_FILE, sep="\t")
    reactome_edges["gene_a"] = reactome_edges["gene_a"].astype(str).str.upper().str.strip()
    reactome_edges["gene_b"] = reactome_edges["gene_b"].astype(str).str.upper().str.strip()

    reactome_genes = sorted(set(reactome_edges["gene_a"]).union(set(reactome_edges["gene_b"])))
    expr_genes = set(expr["gene_symbol"].tolist())
    graph_genes = [g for g in reactome_genes if g in expr_genes]

    expr = expr.set_index("gene_symbol")
    X = expr.loc[graph_genes, sample_ids].T.to_numpy(dtype=np.float32)
    y = (pheno["sample_type"] == args.tumor_label).astype(np.int64).to_numpy()

    feat_to_idx = {g: i for i, g in enumerate(graph_genes)}
    edge_df = reactome_edges[
        reactome_edges["gene_a"].isin(feat_to_idx) & reactome_edges["gene_b"].isin(feat_to_idx)
    ].copy()
    rows = edge_df["gene_a"].map(feat_to_idx).to_numpy()
    cols = edge_df["gene_b"].map(feat_to_idx).to_numpy()
    data = np.ones(len(rows) * 2, dtype=np.float32)
    A = sparse.coo_matrix(
        (data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(len(graph_genes), len(graph_genes)),
    ).tocsr()
    A.data[:] = 1.0
    A = A.sign()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    sparse.save_npz(out_dir / "L.npz", L)
    pheno.assign(y=y).to_csv(out_dir / "sample_table.tsv", sep="\t", index=False)

    with open(out_dir / "feature_graph_symbols.txt", "w", encoding="utf-8") as f:
        for g in graph_genes:
            f.write(g + "\n")

    meta = {
        "dataset_name": args.dataset_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_tumor": int((y == 1).sum()),
        "n_normal": int((y == 0).sum()),
        "n_edges": int(A.nnz // 2),
        "n_components": int(sparse.csgraph.connected_components(A, directed=False, return_labels=False)),
        "tumor_label": args.tumor_label,
        "normal_label": args.normal_label,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved dataset to:", out_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_dataset(parse_args())
