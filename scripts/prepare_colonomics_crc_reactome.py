from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "external" / "gse44076"
OUT_DIR = ROOT / "datasets" / "colonomics_crc_reactome"
REACTOME_EDGE_FILE = ROOT / "datasets" / "tcga_brca" / "reactome_edges_symbol.tsv"


def normalize_gene_symbol(gene: str) -> str:
    return str(gene).strip().upper()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-suffixes", type=str, default="M", help="Comma-separated sample suffixes for class 0, e.g. M or N or M,N")
    p.add_argument("--pos-suffixes", type=str, default="T", help="Comma-separated sample suffixes for class 1")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--dataset-name", type=str, default="colonomics_crc_reactome")
    return p.parse_args()


def build_dataset(neg_suffixes=("M",), pos_suffixes=("T",), out_dir: Path = OUT_DIR, dataset_name: str = "colonomics_crc_reactome"):
    expr_path = RAW_DIR / "CLX_Expression_PC1_2014Nov24.txt"
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr.index = expr.index.map(normalize_gene_symbol)
    expr = expr[expr.index.notna() & (expr.index != "")]

    sample_cols = expr.columns.tolist()
    neg_suffixes = tuple(s.strip().upper() for s in neg_suffixes if str(s).strip())
    pos_suffixes = tuple(s.strip().upper() for s in pos_suffixes if str(s).strip())
    neg_cols = [c for c in sample_cols if any(c.endswith(f"_{s}") for s in neg_suffixes)]
    pos_cols = [c for c in sample_cols if any(c.endswith(f"_{s}") for s in pos_suffixes)]

    keep_cols = neg_cols + pos_cols
    expr = expr[keep_cols].apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()

    reactome_edges = pd.read_csv(REACTOME_EDGE_FILE, sep="\t")
    reactome_edges["gene_a"] = reactome_edges["gene_a"].map(normalize_gene_symbol)
    reactome_edges["gene_b"] = reactome_edges["gene_b"].map(normalize_gene_symbol)

    reactome_genes = sorted(set(reactome_edges["gene_a"]).union(set(reactome_edges["gene_b"])))
    expr_genes = set(expr.index.tolist())
    graph_genes = [g for g in reactome_genes if g in expr_genes]

    expr_graph = expr.loc[graph_genes, keep_cols].copy()
    missing_before = int(expr_graph.isna().sum().sum())
    expr_graph = expr_graph.dropna(axis=0, how="all")
    graph_genes = expr_graph.index.tolist()

    if missing_before > 0:
        # The processed Colonomics matrix has a tiny number of gene-sample gaps.
        # Fill them gene-wise to avoid propagating NaNs into training.
        gene_medians = expr_graph.median(axis=1)
        expr_graph = expr_graph.T.fillna(gene_medians).T

    missing_after = int(expr_graph.isna().sum().sum())
    X = expr_graph.loc[graph_genes, keep_cols].T.to_numpy(dtype=np.float32)
    y = np.array([0] * len(neg_cols) + [1] * len(pos_cols), dtype=np.int64)

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

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    sparse.save_npz(out_dir / "L.npz", L)

    with open(out_dir / "feature_graph_symbols.txt", "w", encoding="utf-8") as f:
        for g in graph_genes:
            f.write(g + "\n")

    sample_table = pd.DataFrame(
        {
            "sample_id": keep_cols,
            "label": y,
            "sample_type": [f"class0_{'_'.join(neg_suffixes).lower()}"] * len(neg_cols) + [f"class1_{'_'.join(pos_suffixes).lower()}"] * len(pos_cols),
        }
    )
    sample_table.to_csv(out_dir / "sample_table.tsv", sep="\t", index=False)

    meta = {
        "dataset_name": dataset_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_class0": int(len(neg_cols)),
        "n_class1": int(len(pos_cols)),
        "neg_suffixes": list(neg_suffixes),
        "pos_suffixes": list(pos_suffixes),
        "n_edges": int(A.nnz // 2),
        "n_components": int(sparse.csgraph.connected_components(A, directed=False, return_labels=False)),
        "missing_values_before_impute": missing_before,
        "missing_values_after_impute": missing_after,
    }
    pd.Series(meta).to_json(out_dir / "meta.json", indent=2)

    print("Saved dataset to:", out_dir)
    print(meta)


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        neg_suffixes=args.neg_suffixes.split(","),
        pos_suffixes=args.pos_suffixes.split(","),
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
    )
