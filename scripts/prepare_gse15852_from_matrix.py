from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def parse_series_matrix(matrix_fp: Path):
    sample_ids = []
    sample_titles = []
    table_lines = []

    in_table = False
    with open(matrix_fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")
                sample_ids = [x.strip().strip('"') for x in parts[1:]]

            elif line.startswith("!Sample_title"):
                parts = line.split("\t")
                sample_titles = [x.strip().strip('"') for x in parts[1:]]

            elif line.startswith("!series_matrix_table_begin"):
                in_table = True

            elif line.startswith("!series_matrix_table_end"):
                in_table = False

            elif in_table:
                table_lines.append(line)

    if not sample_ids:
        raise ValueError("No !Sample_geo_accession found.")
    if not sample_titles:
        raise ValueError("No !Sample_title found.")
    if not table_lines:
        raise ValueError("No matrix table found.")

    header = [x.strip().strip('"') for x in table_lines[0].split("\t")]
    rows = [[y.strip().strip('"') for y in x.split("\t")] for x in table_lines[1:]]
    expr = pd.DataFrame(rows, columns=header)

    expr = expr.rename(columns={expr.columns[0]: "ID_REF"})
    expr.columns = [str(c).strip().strip('"') for c in expr.columns]
    expr["ID_REF"] = expr["ID_REF"].astype(str).str.strip().str.strip('"')

    for c in expr.columns[1:]:
        expr[c] = pd.to_numeric(expr[c].astype(str).str.strip().str.strip('"'), errors="coerce")

    expr = expr.set_index("ID_REF")

    labels = pd.DataFrame({
        "sample_id": sample_ids,
        "sample_title": sample_titles,
    })

    def title_to_label(x: str):
        s = str(x).lower()
        if s.startswith("normal"):
            return 0
        if s.startswith("cancer") or s.startswith("tumor"):
            return 1
        raise ValueError(f"Cannot infer label from sample title: {x}")

    labels["label"] = labels["sample_title"].map(title_to_label)

    # 保证表达矩阵列顺序与 labels 一致
    print("expr first 5 cols:", expr.columns[:5].tolist())
    print("label first 5 ids:", labels["sample_id"].tolist()[:5])
    expr = expr[labels["sample_id"].tolist()]

    return expr, labels


def load_gpl96_annotation(gpl_fp: Path):
    lines = []
    header = None
    data_started = False

    with open(gpl_fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            # 真正表头通常长这样：
            # ID\tGene title\tGene symbol\t...
            if line.startswith("ID\t"):
                header = line.split("\t")
                data_started = True
                continue

            if data_started:
                if not line.strip():
                    continue
                parts = line.split("\t")
                # 只保留列数和表头一致的行
                if len(parts) == len(header):
                    lines.append(parts)

    if header is None:
        raise ValueError("Cannot find GPL96 table header starting with 'ID\\t'.")

    ann = pd.DataFrame(lines, columns=header)
    ann.columns = [str(c).strip() for c in ann.columns]

    probe_col = None
    gene_col = None

    for c in ann.columns:
        cl = c.lower().strip()
        if probe_col is None and cl == "id":
            probe_col = c
        if gene_col is None and cl == "gene symbol":
            gene_col = c

    if probe_col is None:
        raise ValueError(f"Cannot find probe ID column. Columns: {ann.columns.tolist()[:20]}")
    if gene_col is None:
        raise ValueError(f"Cannot find gene symbol column. Columns: {ann.columns.tolist()[:20]}")

    ann = ann[[probe_col, gene_col]].copy()
    ann.columns = ["ID_REF", "gene_symbol"]

    ann["ID_REF"] = ann["ID_REF"].astype(str).str.strip()
    ann["gene_symbol"] = ann["gene_symbol"].astype(str).str.strip()

    # 清理空值、多 symbol、控制探针等
    ann = ann[(ann["gene_symbol"] != "") & (ann["gene_symbol"].str.lower() != "nan")]
    ann = ann[~ann["gene_symbol"].isin(["---", "NA", "na"])]

    ann["gene_symbol"] = (
        ann["gene_symbol"]
        .str.split(r"///|//|;|,")
        .str[0]
        .str.strip()
    )

    ann = ann[(ann["gene_symbol"] != "") & (ann["gene_symbol"].str.lower() != "nan")]
    ann = ann[~ann["gene_symbol"].isin(["---", "NA", "na"])]

    ann = ann.drop_duplicates(subset=["ID_REF"], keep="first")

    print("GPL96 annotation columns:", ann.columns.tolist())
    print("n_annotation_rows =", len(ann))

    return ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_fp", type=str, required=True)
    parser.add_argument("--gpl_fp", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    matrix_fp = Path(args.matrix_fp)
    gpl_fp = Path(args.gpl_fp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_probe, labels = parse_series_matrix(matrix_fp)
    ann = load_gpl96_annotation(gpl_fp)

    common_probes = [p for p in expr_probe.index if p in set(ann["ID_REF"])]
    expr_probe = expr_probe.loc[common_probes].copy()

    ann_map = ann.set_index("ID_REF").loc[common_probes].copy()
    expr_probe["gene_symbol"] = ann_map["gene_symbol"].values

    expr_symbol = expr_probe.groupby("gene_symbol").mean(numeric_only=True)
    expr_symbol.insert(0, "gene_symbol", expr_symbol.index)
    expr_symbol = expr_symbol.reset_index(drop=True)

    expr_out = out_dir / "gse15852_expr_symbol.tsv"
    label_out = out_dir / "gse15852_labels.tsv"

    expr_symbol.to_csv(expr_out, sep="\t", index=False)
    labels[["sample_id", "label"]].to_csv(label_out, sep="\t", index=False)

    print("Saved:")
    print(expr_out)
    print(label_out)
    print()
    print("n_samples =", len(labels))
    print("tumor_count =", int((labels['label'] == 1).sum()))
    print("normal_count =", int((labels['label'] == 0).sum()))
    print("n_probe_rows_after_mapping =", len(common_probes))
    print("n_gene_symbols =", expr_symbol.shape[0])


if __name__ == "__main__":
    main()