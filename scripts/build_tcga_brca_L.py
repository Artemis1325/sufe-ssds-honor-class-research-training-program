from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

edge_path = base / "reactome_edges_symbol_brca.tsv"
feature_path = base / "feature_graph_symbols.txt"

# 读特征顺序
with open(feature_path, "r", encoding="utf-8") as f:
    features = [line.strip() for line in f if line.strip()]

feat_to_idx = {g: i for i, g in enumerate(features)}
p = len(features)

print("==== 1. feature info ====")
print("n_features =", p)
print("first 10 features =", features[:10])

# 读边表
edges = pd.read_csv(edge_path, sep="\t")

print("\n==== 2. raw edges ====")
print("shape =", edges.shape)
print(edges.head())

# 只保留两端都在 feature 里的边
edges = edges[
    edges["gene_a"].isin(feat_to_idx) & edges["gene_b"].isin(feat_to_idx)
].copy()

# 转成索引边
rows = edges["gene_a"].map(feat_to_idx).to_numpy()
cols = edges["gene_b"].map(feat_to_idx).to_numpy()

# 无向图：补对称边
all_rows = np.concatenate([rows, cols])
all_cols = np.concatenate([cols, rows])
data = np.ones(len(all_rows), dtype=np.float32)

A = sparse.coo_matrix((data, (all_rows, all_cols)), shape=(p, p)).tocsr()
A.data[:] = 1.0
A = A.sign()

deg = np.asarray(A.sum(axis=1)).ravel()
D = sparse.diags(deg)
L = (D - A).tocsr()

print("\n==== 3. graph stats ====")
print("A shape =", A.shape)
print("A nnz =", A.nnz)
print("undirected edges =", A.nnz // 2)
print("degree min =", float(deg.min()))
print("degree max =", float(deg.max()))
print("degree mean =", float(deg.mean()))
print("isolated nodes =", int((deg == 0).sum()))

# 保存
sparse.save_npz(base / "L.npz", L)
sparse.save_npz(base / "A.npz", A)

print("\n==== 4. saved ====")
print("saved:", base / "L.npz")
print("saved:", base / "A.npz")