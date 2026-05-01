from pathlib import Path
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

edge_path = base / "reactome_edges_ensembl.tsv"
map_path = base / "ensembl_to_symbol.tsv"
feature_symbol_path = base / "feature_symbols.txt"

# 读边表
edges = pd.read_csv(edge_path, sep="\t")

# 读映射表
emap = pd.read_csv(map_path, sep="\t")
emap = emap[["ensembl_id", "gene_symbol", "gene_type"]].copy()

# 只保留 protein_coding 且 gene_symbol 非空
emap = emap[
    (emap["gene_type"] == "protein_coding") &
    (emap["gene_symbol"].notna())
].copy()

# 去重，避免一个 ensembl 多行
emap = emap.drop_duplicates(subset=["ensembl_id"]).reset_index(drop=True)

print("==== 1. 输入数据 ====")
print("edges shape =", edges.shape)
print("emap shape =", emap.shape)

# 左右端点分别映射
edges2 = edges.merge(
    emap.rename(columns={"ensembl_id": "ensembl_a", "gene_symbol": "gene_a"}),
    on="ensembl_a",
    how="left",
)
edges2 = edges2.merge(
    emap.rename(columns={"ensembl_id": "ensembl_b", "gene_symbol": "gene_b"}),
    on="ensembl_b",
    how="left",
)

print("\n==== 2. 映射后 ====")
print("shape =", edges2.shape)
print(edges2.head())

# 去掉未映射
edges2 = edges2[edges2["gene_a"].notna() & edges2["gene_b"].notna()].copy()

# 去掉自环
edges2 = edges2[edges2["gene_a"] != edges2["gene_b"]].copy()

# 排序标准化无向边
ab = edges2[["gene_a", "gene_b"]].astype(str)
gene_min = ab.min(axis=1)
gene_max = ab.max(axis=1)
edges2["gene_a"] = gene_min
edges2["gene_b"] = gene_max

# 去重
edges2 = edges2[["gene_a", "gene_b", "edge_type"]].drop_duplicates().reset_index(drop=True)

print("\n==== 3. gene symbol 边表 ====")
print("shape =", edges2.shape)
print(edges2.head(10))

# 读 BRCA 表达特征
with open(feature_symbol_path, "r", encoding="utf-8") as f:
    feat_symbols = [line.strip() for line in f if line.strip()]
feat_set = set(feat_symbols)

print("\n==== 4. BRCA 特征集 ====")
print("n_feature_symbols =", len(feat_symbols))

# 与 BRCA 特征取交集
edges3 = edges2[
    edges2["gene_a"].isin(feat_set) & edges2["gene_b"].isin(feat_set)
].copy().reset_index(drop=True)

print("\n==== 5. 与 BRCA 特征取交集后 ====")
print("shape =", edges3.shape)
print(edges3.head(10))

graph_genes = sorted(set(edges3["gene_a"]).union(set(edges3["gene_b"])))
print("\n==== 6. 图中基因数 ====")
print("n_graph_genes =", len(graph_genes))
print("graph gene preview:", graph_genes[:20])

# 保存
edges2.to_csv(base / "reactome_edges_symbol.tsv", sep="\t", index=False)
edges3.to_csv(base / "reactome_edges_symbol_brca.tsv", sep="\t", index=False)

with open(base / "reactome_graph_genes_brca.txt", "w", encoding="utf-8") as f:
    for g in graph_genes:
        f.write(g + "\n")

print("\n==== 7. 已保存 ====")
print("saved:", base / "reactome_edges_symbol.tsv")
print("saved:", base / "reactome_edges_symbol_brca.tsv")
print("saved:", base / "reactome_graph_genes_brca.txt")