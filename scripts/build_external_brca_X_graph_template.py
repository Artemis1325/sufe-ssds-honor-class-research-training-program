from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def read_gene_list(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def make_unique_sample_names(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr_tsv", type=str, required=True,
                        help="External expression matrix TSV: rows=genes, cols=samples, first col=gene symbol")
    parser.add_argument("--label_tsv", type=str, required=True,
                        help="TSV with at least two columns: sample_id, label (tumor=1, normal=0)")
    parser.add_argument("--graph_symbols", type=str,
                        default="datasets/tcga_brca/feature_graph_symbols.txt")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    expr_fp = Path(args.expr_tsv)
    label_fp = Path(args.label_tsv)
    graph_fp = Path(args.graph_symbols)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取外部表达矩阵
    expr = pd.read_csv(expr_fp, sep="\t")
    if expr.shape[1] < 2:
        raise ValueError("Expression TSV must have at least 2 columns.")

    gene_col = expr.columns[0]
    expr[gene_col] = expr[gene_col].astype(str).str.strip()

    # 去掉空 gene symbol
    expr = expr[expr[gene_col].notna() & (expr[gene_col] != "")]
    # 对重复 gene symbol 取均值
    expr = expr.groupby(gene_col, as_index=True).mean(numeric_only=True)

    # 保证 sample 列名唯一
    expr.columns = make_unique_sample_names(expr.columns.tolist())

    # 2) 读取标签
    labels = pd.read_csv(label_fp, sep="\t")
    if "sample_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError("label_tsv must contain columns: sample_id, label")
    labels["sample_id"] = labels["sample_id"].astype(str)

    # 只保留标签里有、且表达矩阵里也有的样本
    common_samples = [s for s in labels["sample_id"].tolist() if s in expr.columns]
    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between expression matrix and label file.")

    labels = labels.set_index("sample_id").loc[common_samples].reset_index()
    expr = expr[common_samples]

    # 3) 对齐到 TCGA 的 graph gene 顺序
    graph_symbols = read_gene_list(graph_fp)
    kept_genes = [g for g in graph_symbols if g in expr.index]
    missing_genes = [g for g in graph_symbols if g not in expr.index]

    X = np.zeros((len(common_samples), len(graph_symbols)), dtype=np.float32)
    gene_to_row = {g: i for i, g in enumerate(expr.index.tolist())}

    for j, g in enumerate(graph_symbols):
        if g in gene_to_row:
            X[:, j] = expr.loc[g, common_samples].to_numpy(dtype=np.float32)

    y = labels["label"].to_numpy(dtype=np.int64)

    # 4) 保存
    np.save(out_dir / "X_graph.npy", X)
    np.save(out_dir / "y.npy", y)

    with open(out_dir / "feature_graph_symbols.txt", "w", encoding="utf-8") as f:
        for g in graph_symbols:
            f.write(g + "\n")

    labels.to_csv(out_dir / "sample_table.tsv", sep="\t", index=False)

    with open(out_dir / "present_genes.txt", "w", encoding="utf-8") as f:
        for g in kept_genes:
            f.write(g + "\n")

    with open(out_dir / "missing_genes.txt", "w", encoding="utf-8") as f:
        for g in missing_genes:
            f.write(g + "\n")

    print("==== external dataset aligned ====")
    print("n_samples =", len(common_samples))
    print("X_graph shape =", X.shape)
    print("y shape =", y.shape)
    print("n_graph_genes =", len(graph_symbols))
    print("n_present_genes =", len(kept_genes))
    print("n_missing_genes =", len(missing_genes))
    print("gene_coverage =", len(kept_genes) / len(graph_symbols))
    print("tumor_count =", int((y == 1).sum()))
    print("normal_count =", int((y == 0).sum()))
    print("saved to:", out_dir)


if __name__ == "__main__":
    main()