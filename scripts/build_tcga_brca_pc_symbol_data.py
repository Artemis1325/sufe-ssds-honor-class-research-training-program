from pathlib import Path
import numpy as np
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

x_path = base / "X.npy"
sample_table_path = base / "sample_table.tsv"
feature_map_path = base / "feature_symbol_mapping.tsv"

# 读取
X = np.load(x_path)
sample_table = pd.read_csv(sample_table_path, sep="\t")
feat_map = pd.read_csv(feature_map_path, sep="\t")

print("==== 1. 原始数据 ====")
print("X shape =", X.shape)
print("sample_table shape =", sample_table.shape)
print("feature_map shape =", feat_map.shape)

# 只保留 protein_coding 且 gene_symbol 非空
feat_keep = feat_map[
    (feat_map["gene_type"] == "protein_coding") &
    (feat_map["gene_symbol"].notna())
].copy().reset_index(drop=True)

print("\n==== 2. 初筛后特征 ====")
print("n_kept_before_dedup =", len(feat_keep))
print(feat_keep.head())

# 处理 gene_symbol 重复：同一个 symbol 可能对应多个 Ensembl
dup_counts = feat_keep["gene_symbol"].value_counts()
dup_symbols = dup_counts[dup_counts > 1].index.tolist()

print("\n==== 3. gene symbol 重复情况 ====")
print("n_duplicate_symbols =", len(dup_symbols))
print("top duplicated symbols:")
print(dup_counts.head(20))

# 先给每个保留特征加原始列索引
feat_keep["orig_idx"] = feat_keep.index

# 注意：上面 reset_index 后丢了原始位置，所以这里重新来一遍
feat_keep = feat_map.copy()
feat_keep["orig_idx"] = np.arange(len(feat_keep))
feat_keep = feat_keep[
    (feat_keep["gene_type"] == "protein_coding") &
    (feat_keep["gene_symbol"].notna())
].copy()

# 取对应列
X_pc = X[:, feat_keep["orig_idx"].to_numpy()]

# 变成 dataframe 便于按 symbol 聚合
expr_df = pd.DataFrame(X_pc, columns=feat_keep["gene_symbol"].tolist())

# 对重复 gene symbol 取均值
expr_df = expr_df.groupby(level=0, axis=1).mean()

feature_symbols = expr_df.columns.tolist()
X_symbol = expr_df.to_numpy(dtype=np.float32)

print("\n==== 4. 聚合后数据 ====")
print("X_symbol shape =", X_symbol.shape)
print("n_final_symbols =", len(feature_symbols))
print("first 10 symbols =", feature_symbols[:10])

# 保存
np.save(base / "X_pc_symbol.npy", X_symbol)

with open(base / "feature_symbols.txt", "w", encoding="utf-8") as f:
    for g in feature_symbols:
        f.write(g + "\n")

# 另存一个保留映射表，便于追溯
feat_keep.to_csv(base / "feature_symbol_mapping_protein_coding.tsv", sep="\t", index=False)

print("\n==== 5. 已保存 ====")
print("saved:", base / "X_pc_symbol.npy")
print("saved:", base / "feature_symbols.txt")
print("saved:", base / "feature_symbol_mapping_protein_coding.tsv")

print("\n==== 6. 基本检查 ====")
print("n_samples =", X_symbol.shape[0])
print("n_features =", X_symbol.shape[1])
print("first row first 10 values =", X_symbol[0, :10].tolist())