from pathlib import Path
import pandas as pd
import numpy as np

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

expr_path = base / "TCGA-BRCA.star_tpm.tsv"
sample_table_path = base / "sample_table.tsv"

# 读样本表，确定保留样本及顺序
sample_table = pd.read_csv(sample_table_path, sep="\t")
sample_ids = sample_table["sample_id"].tolist()

print("==== 1. sample table ====")
print("n_samples =", len(sample_ids))
print("sample preview:", sample_ids[:5])

# 读 expression
expr = pd.read_csv(expr_path, sep="\t")

print("\n==== 2. raw expression ====")
print("raw shape =", expr.shape)
print("columns preview:", expr.columns[:5].tolist())

# 第一列基因ID
gene_col = expr.columns[0]
expr = expr.rename(columns={gene_col: "gene_id"})

# 只保留 gene_id + sample_ids
expr = expr[["gene_id"] + sample_ids].copy()

print("\n==== 3. subset expression ====")
print("subset shape =", expr.shape)
print(expr.iloc[:5, :5])

# 去掉 Ensembl 版本号，例如 ENSG00000000003.15 -> ENSG00000000003
expr["gene_id"] = expr["gene_id"].astype(str).str.split(".").str[0]

# 若有重复 gene_id，取均值
expr = expr.groupby("gene_id", as_index=False).mean(numeric_only=True)

print("\n==== 4. after removing version / dedup ====")
print("shape =", expr.shape)
print(expr.iloc[:5, :5])

# 转成 X: 行是样本，列是基因
feature_names = expr["gene_id"].tolist()
X = expr[sample_ids].T.to_numpy(dtype=np.float32)

# 保存
np.save(base / "X.npy", X)

with open(base / "feature_ensembl_ids.txt", "w", encoding="utf-8") as f:
    for g in feature_names:
        f.write(g + "\n")

print("\n==== 5. saved ====")
print("saved:", base / "X.npy")
print("saved:", base / "feature_ensembl_ids.txt")

print("\n==== 6. final X info ====")
print("X shape =", X.shape)
print("n_features =", len(feature_names))
print("first 5 genes =", feature_names[:5])
print("first row first 5 values =", X[0, :5].tolist())