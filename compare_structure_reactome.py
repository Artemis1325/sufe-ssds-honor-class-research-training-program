"""
import pandas as pd
import networkx as nx
from pathlib import Path

base = Path("datasets/tcga_brca")
res0 = Path("results/brca_sanity_g0/selected_genes_seed42.txt")
res1 = Path("results/brca_sanity_g0p001/selected_genes_seed42.txt")
edge_path = base / "reactome_edges_symbol_brca.tsv"

# 读边表
df = pd.read_csv(edge_path, sep="\t")
edges = list(df[["gene_a", "gene_b"]].itertuples(index=False, name=None))

# 构图
G = nx.Graph()
G.add_edges_from(edges)

def load_gene_set(path):
    return {line.strip() for line in open(path, "r", encoding="utf-8") if line.strip()}

def summarize_subgraph(gene_set, name):
    H = G.subgraph(gene_set).copy()
    n = H.number_of_nodes()
    m = H.number_of_edges()
    degrees = dict(H.degree())
    avg_degree = (sum(degrees.values()) / n) if n > 0 else 0.0
    num_isolates = sum(1 for _, d in degrees.items() if d == 0)
    comps = list(nx.connected_components(H))
    num_components = len(comps)
    largest_cc = max((len(c) for c in comps), default=0)
    density = nx.density(H) if n > 1 else 0.0

    print(f"===== {name} =====")
    print("nodes:", n)
    print("edges:", m)
    print("avg_degree:", avg_degree)
    print("isolates:", num_isolates)
    print("components:", num_components)
    print("largest_cc:", largest_cc)
    print("density:", density)
    print()

g0 = load_gene_set(res0)
g1 = load_gene_set(res1)

summarize_subgraph(g0, "gamma=0")
summarize_subgraph(g1, "gamma=0.001")
"""

import numpy as np
import pandas as pd
from pathlib import Path

base = Path("datasets/tcga_brca")

theta0_path = Path(r"results/brca_sanity_g0/theta_20260316-214326_seed42.npy")
theta1_path = Path(r"results/brca_sanity_g0p001/theta_20260316-214810_seed42.npy")

sel0_path = Path(r"results/brca_sanity_g0/selected_20260316-214326_seed42.npy")
sel1_path = Path(r"results/brca_sanity_g0p001/selected_20260316-214810_seed42.npy")

gene_path = base / "feature_graph_symbols.txt"
edge_path = base / "reactome_edges_symbol_brca.tsv"

# gene order
with open(gene_path, "r", encoding="utf-8") as f:
    genes = [line.strip() for line in f if line.strip()]
gene_to_idx = {g: i for i, g in enumerate(genes)}

# load arrays
theta0 = np.load(theta0_path)
theta1 = np.load(theta1_path)
sel0 = np.load(sel0_path)
sel1 = np.load(sel1_path)

# edges -> index pairs
df = pd.read_csv(edge_path, sep="\t")
pairs = []
for a, b in df[["gene_a", "gene_b"]].itertuples(index=False, name=None):
    if a in gene_to_idx and b in gene_to_idx:
        pairs.append((gene_to_idx[a], gene_to_idx[b]))

pairs = np.array(pairs, dtype=np.int64)
i = pairs[:, 0]
j = pairs[:, 1]

def summarize(theta, sel, name):
    diff = np.abs(theta[i] - theta[j])

    both_selected = (sel[i] == 1) & (sel[j] == 1)
    diff_sel = diff[both_selected]

    print(f"===== {name} =====")
    print("num graph edges used:", len(diff))
    print("all-edge mean |theta_i-theta_j|:", float(diff.mean()))
    print("all-edge median |theta_i-theta_j|:", float(np.median(diff)))
    print("all-edge p90 |theta_i-theta_j|:", float(np.quantile(diff, 0.9)))
    print("selected-selected edges:", int(both_selected.sum()))
    if len(diff_sel) > 0:
        print("selected-edge mean |theta_i-theta_j|:", float(diff_sel.mean()))
        print("selected-edge median |theta_i-theta_j|:", float(np.median(diff_sel)))
        print("selected-edge p90 |theta_i-theta_j|:", float(np.quantile(diff_sel, 0.9)))
    else:
        print("selected-edge mean |theta_i-theta_j|: NA")
        print("selected-edge median |theta_i-theta_j|: NA")
        print("selected-edge p90 |theta_i-theta_j|: NA")
    print()

summarize(theta0, sel0, "gamma=0")
summarize(theta1, sel1, "gamma=0.001")