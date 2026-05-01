from pathlib import Path
import numpy as np
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

x_path = base / "X_pc_symbol.npy"
feature_symbol_path = base / "feature_symbols.txt"
graph_gene_path = base / "reactome_graph_genes_brca.txt"

# 读表达矩阵
X = np.load(x_path)

with open(feature_symbol_path, "r", encoding="utf-8") as f:
    feature_symbols = [line.strip() for line in f if line.strip()]

with open(graph_gene_path, "r", encoding="utf-8") as f:
    graph_genes = [line.strip() for line in f if line.strip()]

print("==== 1. 输入 ====")
print("X shape =", X.shape)
print("n_feature_symbols =", len(feature_symbols))
print("n_graph_genes =", len(graph_genes))

# 建立 feature symbol -> 列索引
feat_to_idx = {g: i for i, g in enumerate(feature_symbols)}

# 按 graph_genes 顺序取列
kept_genes = [g for g in graph_genes if g in feat_to_idx]
kept_idx = [feat_to_idx[g] for g in kept_genes]

X_graph = X[:, kept_idx].astype(np.float32)

print("\n==== 2. 输出矩阵 ====")
print("X_graph shape =", X_graph.shape)
print("first 10 genes =", kept_genes[:10])
print("first row first 10 values =", X_graph[0, :10].tolist())

# 保存
np.save(base / "X_graph.npy", X_graph)

with open(base / "feature_graph_symbols.txt", "w", encoding="utf-8") as f:
    for g in kept_genes:
        f.write(g + "\n")

print("\n==== 3. 已保存 ====")
print("saved:", base / "X_graph.npy")
print("saved:", base / "feature_graph_symbols.txt")