from pathlib import Path
import argparse
import gzip
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
REACTOME_EDGE_FILE = ROOT / "datasets" / "tcga_brca" / "reactome_edges_symbol.tsv"


def compile_patterns(raw: str):
    return [re.compile(x, flags=re.I) for x in raw.split("|||") if x.strip()]


def parse_series_matrix(matrix_fp: Path):
    sample_meta = {}
    table_lines = []
    in_table = False

    opener = gzip.open if matrix_fp.suffix == ".gz" else open
    with opener(matrix_fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_"):
                key = line.split("\t", 1)[0][8:]
                values = [x.strip().strip('"') for x in line.split("\t")[1:]]
                sample_meta[key] = values
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

    n = len(sample_meta["geo_accession"])
    labels = pd.DataFrame({"sample_id": sample_meta["geo_accession"]})
    for k, vals in sample_meta.items():
        if len(vals) == n:
            labels[f"sample_{k}"] = vals
    expr = expr[labels["sample_id"].tolist()]
    return expr, labels


def load_gpl_annotation(gpl_fp: Path):
    lines = []
    header = None
    data_started = False
    opener = gzip.open if gpl_fp.suffix == ".gz" else open

    with opener(gpl_fp, "rt", encoding="utf-8", errors="ignore") as f:
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
    symbol_col = None
    for c in ann.columns:
        if c.lower().strip() == "gene symbol":
            symbol_col = c
            break
    if symbol_col is None:
        raise ValueError(f"Cannot find gene symbol column in {gpl_fp}")

    ann = ann[["ID", symbol_col]].copy()
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


def infer_labels(labels: pd.DataFrame, pos_patterns, neg_patterns):
    text_cols = [c for c in labels.columns if c.startswith("sample_")]
    combined = labels[text_cols].fillna("").astype(str).agg(" ||| ".join, axis=1)

    def match_any(s, patterns):
        return any(p.search(s) for p in patterns)

    y = []
    keep = []
    for s in combined:
        is_pos = match_any(s, pos_patterns)
        is_neg = match_any(s, neg_patterns)
        if is_pos and not is_neg:
            y.append(1)
            keep.append(True)
        elif is_neg and not is_pos:
            y.append(0)
            keep.append(True)
        else:
            y.append(-1)
            keep.append(False)

    out = labels.copy()
    out["label"] = y
    out = out[np.array(keep, dtype=bool)].reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-fp", required=True)
    p.add_argument("--gpl-fp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--positive-patterns", required=True, help="regexes joined by |||")
    p.add_argument("--negative-patterns", required=True, help="regexes joined by |||")
    args = p.parse_args()

    expr_probe, labels = parse_series_matrix(Path(args.matrix_fp))
    ann = load_gpl_annotation(Path(args.gpl_fp))
    labels = infer_labels(labels, compile_patterns(args.positive_patterns), compile_patterns(args.negative_patterns))

    sample_ids = labels["sample_id"].tolist()
    expr_probe = expr_probe[sample_ids]
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    sparse.save_npz(out_dir / "L.npz", L)
    labels.to_csv(out_dir / "sample_table.tsv", sep="\t", index=False)
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
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
