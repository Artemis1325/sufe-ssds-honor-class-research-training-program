"""
import numpy as np
from pathlib import Path

base = Path("datasets/tcga_brca")
res = Path("results/brca_sanity_g0p001")

# 读入基因名
with open(base / "feature_graph_symbols.txt", "r", encoding="utf-8") as f:
    genes = [line.strip() for line in f if line.strip()]

# 读入 selected mask
sel = np.load(res / "selected_20260316-214810_seed42.npy")

# 导出选中基因
selected_genes = [g for g, s in zip(genes, sel) if s == 1]

out_path = res / "selected_genes_seed42.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for g in selected_genes:
        f.write(g + "\n")

print("num selected genes:", len(selected_genes))
print("first 20 selected genes:")
for g in selected_genes[:20]:
    print(g)

print("saved to:", out_path)
"""

import numpy as np
from pathlib import Path

base = Path("datasets/tcga_brca")
res = Path("results/brca_sanity_g0")

with open(base / "feature_graph_symbols.txt", "r", encoding="utf-8") as f:
    genes = [line.strip() for line in f if line.strip()]

sel = np.load(res / "selected_20260316-214326_seed42.npy")

selected_genes = [g for g, s in zip(genes, sel) if s == 1]

out_path = res / "selected_genes_seed42.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for g in selected_genes:
        f.write(g + "\n")

print("num selected genes:", len(selected_genes))
print("first 20 selected genes:")
for g in selected_genes[:20]:
    print(g)

print("saved to:", out_path)