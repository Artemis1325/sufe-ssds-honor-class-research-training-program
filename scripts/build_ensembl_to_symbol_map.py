from pathlib import Path
import re
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")
gtf_path = base / "gencode.v49.basic.annotation.gtf"

def parse_attr(attr_str: str, key: str):
    m = re.search(rf'{key} "([^"]+)"', attr_str)
    return m.group(1) if m else None

rows = []

with open(gtf_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 9:
            continue

        chrom, source, feature, start, end, score, strand, frame, attrs = parts

        # 只取 gene 行
        if feature != "gene":
            continue

        gene_id = parse_attr(attrs, "gene_id")
        gene_name = parse_attr(attrs, "gene_name")
        gene_type = parse_attr(attrs, "gene_type")

        if gene_id is None or gene_name is None:
            continue

        # 去掉版本号
        gene_id = gene_id.split(".")[0]

        rows.append((gene_id, gene_name, gene_type))

df = pd.DataFrame(rows, columns=["ensembl_id", "gene_symbol", "gene_type"])
df = df.drop_duplicates(subset=["ensembl_id"]).reset_index(drop=True)

out_path = base / "ensembl_to_symbol.tsv"
df.to_csv(out_path, sep="\t", index=False)

print("saved:", out_path)
print("shape =", df.shape)
print(df.head(10))
print()
print("gene_type top 20:")
print(df["gene_type"].value_counts().head(20))