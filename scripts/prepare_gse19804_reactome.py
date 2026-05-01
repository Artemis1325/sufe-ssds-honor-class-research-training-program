from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "external" / "gse19804"
OUT_DIR = ROOT / "datasets" / "gse19804_luad_reactome"
GPL_FP = RAW_DIR / "GPL570.annot.gz"
MATRIX_FP = RAW_DIR / "GSE19804_series_matrix.txt.gz"
REACTOME_EDGE_FILE = ROOT / "datasets" / "tcga_brca" / "reactome_edges_symbol.tsv"


def parse_series_matrix(matrix_fp: Path):
    sample_ids = []
    sample_titles = []
    table_lines = []
    in_table = False

    with gzip.open(matrix_fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip().strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                sample_titles = [x.strip().strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)

    header = [x.strip().strip('"') for x in table_lines[0].split("\t")]
    rows = [[y.strip().strip('"') for y in x.split("\t")] for x in table_lines[1:]]
    expr = pd.DataFrame(rows, columns=header)
    expr = expr.rename(columns={expr.columns[0]: "ID_REF"})
    for c in expr.columns[1:]:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    expr = expr.set_index("ID_REF")

    labels = pd.DataFrame({"sample_id": sample_ids, "sample_title": sample_titles})
    labels["label"] = labels["sample_title"].map(lambda s: 1 if str(s).endswith("T") else 0)
    expr = expr[labels["sample_id"].tolist()]
    return expr, labels


def load_gpl570_annotation(gpl_fp: Path):
    lines = []
    header = None
    data_started = False

    with gzip.open(gpl_fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("ID\t"):
                header = line.split("\t")
                data_started = True
                continue
            if data_started and line.strip():
                parts = line.split("\t")
                if len(parts) == len(header):
                    lines.append(parts)

    ann = pd.DataFrame(lines, columns=header)
    ann = ann[["ID", "Gene symbol"]].copy()
    ann.columns = ["ID_REF", "gene_symbol"]
    ann["gene_symbol"] = (
        ann["gene_symbol"]
        .astype(str)
        .str.strip()
        .str.split(r"///|//|;|,")
        .str[0]
        .str.strip()
        .str.upper()
    )
    ann = ann[(ann["gene_symbol"] != "") & ~ann["gene_symbol"].isin(["---", "NA", "NAN"])]
    ann = ann.drop_duplicates(subset=["ID_REF"], keep="first")
    return ann


def main():
    expr_probe, labels = parse_series_matrix(MATRIX_FP)
    ann = load_gpl570_annotation(GPL_FP)

    common_probes = [p for p in expr_probe.index if p in set(ann["ID_REF"])]
    expr_probe = expr_probe.loc[common_probes].copy()
    ann_map = ann.set_index("ID_REF").loc[common_probes].copy()
    expr_probe["gene_symbol"] = ann_map["gene_symbol"].values
    expr_symbol = expr_probe.groupby("gene_symbol").mean(numeric_only=True)

    reactome_edges = pd.read_csv(REACTOME_EDGE_FILE, sep="\t")
    reactome_edges["gene_a"] = reactome_edges["gene_a"].astype(str).str.upper().str.strip()
    reactome_edges["gene_b"] = reactome_edges["gene_b"].astype(str).str.upper().str.strip()
    reactome_genes = sorted(set(reactome_edges["gene_a"]).union(set(reactome_edges["gene_b"])))
    graph_genes = [g for g in reactome_genes if g in expr_symbol.index]

    sample_ids = labels["sample_id"].tolist()
    X = expr_symbol.loc[graph_genes, sample_ids].T.to_numpy(dtype=np.float32)
    y = labels["label"].to_numpy(dtype=np.int64)

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    sparse.save_npz(OUT_DIR / "L.npz", L)
    labels.assign(sample_type=np.where(labels["label"] == 1, "tumor", "adjacent_normal")).to_csv(
        OUT_DIR / "sample_table.tsv", sep="\t", index=False
    )
    with open(OUT_DIR / "feature_graph_symbols.txt", "w", encoding="utf-8") as f:
        for g in graph_genes:
            f.write(g + "\n")

    meta = {
        "dataset_name": "gse19804_luad_reactome",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_tumor": int((y == 1).sum()),
        "n_normal": int((y == 0).sum()),
        "n_edges": int(A.nnz // 2),
        "n_components": int(sparse.csgraph.connected_components(A, directed=False, return_labels=False)),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved dataset to:", OUT_DIR)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
