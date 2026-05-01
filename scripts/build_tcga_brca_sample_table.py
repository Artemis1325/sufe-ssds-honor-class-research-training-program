from pathlib import Path
import pandas as pd
import numpy as np

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

expr_path = base / "TCGA-BRCA.star_tpm.tsv"
clinical_path = base / "TCGA-BRCA.clinical.tsv"

# 读 expression，拿样本顺序
expr = pd.read_csv(expr_path, sep="\t")
expr_samples = expr.columns[1:].tolist()

# 读 clinical
clinical = pd.read_csv(clinical_path, sep="\t")

# 只保留我们需要的列
df = clinical[["sample", "sample_type.samples", "tissue_type.samples"]].copy()
df = df.rename(columns={
    "sample": "sample_id",
    "sample_type.samples": "sample_type",
    "tissue_type.samples": "tissue_type",
})

# 只保留 expression 中实际存在的样本，并按 expression 顺序重排
df = df[df["sample_id"].isin(expr_samples)].copy()
df["sample_id"] = pd.Categorical(df["sample_id"], categories=expr_samples, ordered=True)
df = df.sort_values("sample_id").reset_index(drop=True)

# 只保留二分类样本
keep_types = ["Primary Tumor", "Solid Tissue Normal"]
df = df[df["sample_type"].isin(keep_types)].copy().reset_index(drop=True)

# 生成标签：Tumor=1, Normal=0
df["y"] = (df["sample_type"] == "Primary Tumor").astype(np.int64)

# 保存样本表
sample_table_path = base / "sample_table.tsv"
df.to_csv(sample_table_path, sep="\t", index=False)

# 保存 y
y = df["y"].to_numpy(dtype=np.int64)
np.save(base / "y.npy", y)

# 保存 sample ids
with open(base / "sample_ids.txt", "w", encoding="utf-8") as f:
    for s in df["sample_id"]:
        f.write(str(s) + "\n")

print("saved:", sample_table_path)
print("saved:", base / "y.npy")
print("saved:", base / "sample_ids.txt")
print()
print("shape =", df.shape)
print(df.head())
print()
print(df["sample_type"].value_counts())
print()
print("y sum =", int(df["y"].sum()))
print("y len =", len(df))
print("first 10 y =", y[:10].tolist())