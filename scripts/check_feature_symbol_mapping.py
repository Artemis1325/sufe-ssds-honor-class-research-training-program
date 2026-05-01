from pathlib import Path
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

feature_path = base / "feature_ensembl_ids.txt"
mapping_path = base / "ensembl_to_symbol.tsv"

# 读 feature ensembl ids
with open(feature_path, "r", encoding="utf-8") as f:
    feature_ids = [line.strip() for line in f if line.strip()]

feat_df = pd.DataFrame({"ensembl_id": feature_ids})

# 读映射表
map_df = pd.read_csv(mapping_path, sep="\t")

# 合并
merged = feat_df.merge(map_df, on="ensembl_id", how="left")

print("==== 1. feature 总数 ====")
print("n_features =", len(merged))

print("\n==== 2. 映射成功情况 ====")
n_mapped = merged["gene_symbol"].notna().sum()
print("n_mapped =", int(n_mapped))
print("mapping_rate =", float(n_mapped / len(merged)))

print("\n==== 3. 前10个映射结果 ====")
print(merged.head(10))

print("\n==== 4. 未映射前20个 ====")
unmapped = merged[merged["gene_symbol"].isna()].copy()
print(unmapped.head(20))

print("\n==== 5. gene_type 分布（前20） ====")
print(merged["gene_type"].value_counts(dropna=False).head(20))

# 只看 protein_coding
pc = merged[merged["gene_type"] == "protein_coding"].copy()
print("\n==== 6. protein_coding 情况 ====")
print("n_protein_coding =", len(pc))
print(pc.head(10))

# 保存完整映射结果
out_path = base / "feature_symbol_mapping.tsv"
merged.to_csv(out_path, sep="\t", index=False)
print("\nsaved:", out_path)