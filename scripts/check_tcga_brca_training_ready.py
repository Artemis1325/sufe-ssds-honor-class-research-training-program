from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

X = np.load(base / "X_graph.npy")
y = np.load(base / "y.npy")
L = sparse.load_npz(base / "L.npz")

sample_table = pd.read_csv(base / "sample_table.tsv", sep="\t")

with open(base / "feature_graph_symbols.txt", "r", encoding="utf-8") as f:
    features = [line.strip() for line in f if line.strip()]

print("==== 1. shape 检查 ====")
print("X shape =", X.shape)
print("y shape =", y.shape)
print("L shape =", L.shape)
print("n_features file =", len(features))
print("sample_table shape =", sample_table.shape)

print("\n==== 2. 维度一致性 ====")
print("X rows == y len:", X.shape[0] == len(y))
print("X cols == L dim:", X.shape[1] == L.shape[0] == L.shape[1])
print("X cols == len(features):", X.shape[1] == len(features))
print("X rows == sample_table rows:", X.shape[0] == len(sample_table))

print("\n==== 3. 标签分布 ====")
print("y sum =", int(y.sum()))
print("y len =", len(y))
print("n_neg =", int((y == 0).sum()))
print("n_pos =", int((y == 1).sum()))

print("\n==== 4. 数值基本检查 ====")
print("X dtype =", X.dtype)
print("y dtype =", y.dtype)
print("X finite =", np.isfinite(X).all())
print("y finite =", np.isfinite(y).all())
print("X min =", float(X.min()))
print("X max =", float(X.max()))
print("X mean =", float(X.mean()))
print("X std =", float(X.std()))

print("\n==== 5. 图基本检查 ====")
deg = np.asarray((L.diagonal())).ravel()
print("L nnz =", L.nnz)
print("degree min =", float(deg.min()))
print("degree max =", float(deg.max()))
print("degree mean =", float(deg.mean()))
print("isolated nodes =", int((deg == 0).sum()))

print("\n==== 6. 前几项预览 ====")
print("first 10 y =", y[:10].tolist())
print("first 10 features =", features[:10])
print("first sample id =", sample_table.iloc[0]['sample_id'])
print("first row first 10 X =", X[0, :10].tolist())