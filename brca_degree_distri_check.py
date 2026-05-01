import numpy as np
import scipy.sparse as sp

A = sp.load_npz(r"datasets/tcga_brca/A.npz").tocsr()
deg = np.asarray(A.sum(axis=1)).ravel()

print("num_nodes =", A.shape[0])
print("num_edges =", sp.triu(A, k=1).nnz)
print("min_deg =", deg.min())
print("max_deg =", deg.max())
print("mean_deg =", deg.mean())
print("median_deg =", np.median(deg))
print("num_zero_deg =", (deg == 0).sum())

qs = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
print("degree_quantiles =", {q: float(np.quantile(deg, q)) for q in qs})