import os
import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "tcga_brca")
CMP_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")

# ---------- load graph ----------
A_path = os.path.join(DATA_DIR, "A.npz")
A = sparse.load_npz(A_path).tocsr()
deg = np.asarray(A.sum(axis=1)).ravel()

# ---------- load symbols ----------
symbol_file = os.path.join(DATA_DIR, "feature_graph_symbols.txt")
with open(symbol_file, "r", encoding="utf-8") as f:
    symbols = [x.strip() for x in f if x.strip()]

assert len(symbols) == A.shape[0], (len(symbols), A.shape)

gene_to_idx = {g: i for i, g in enumerate(symbols)}

# ---------- helpers ----------
def read_gene_list(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def read_bias_table(fp):
    # rank index gene real_freq degnull_freq diff_real_minus_degnull
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            rank, idx, gene, real_f, deg_f, diff = parts
            rows.append({
                "rank": int(rank),
                "idx": int(idx),
                "gene": gene,
                "real_freq": int(real_f),
                "deg_freq": int(deg_f),
                "diff": int(diff),
            })
    return rows

def summarize_group(name, idxs):
    vals = deg[idxs]
    q = np.percentile(vals, [0, 25, 50, 75, 100])
    print(f"[{name}] n={len(idxs)}")
    print(f"  mean={vals.mean():.3f}")
    print(f"  std ={vals.std(ddof=0):.3f}")
    print(f"  min / q1 / median / q3 / max = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} / {q[3]:.1f} / {q[4]:.1f}")
    print()

# ---------- load gene sets ----------
real_only = read_gene_list(os.path.join(CMP_DIR, "real_only_minfreq5.txt"))
deg_only  = read_gene_list(os.path.join(CMP_DIR, "degnull_only_minfreq5.txt"))
shared    = read_gene_list(os.path.join(CMP_DIR, "shared_minfreq5.txt"))

real_biased = read_bias_table(os.path.join(CMP_DIR, "real_biased_diff_ge_5.tsv"))
deg_biased  = read_bias_table(os.path.join(CMP_DIR, "degnull_biased_diff_ge_5.tsv"))

# top stronger-biased subsets
real_top50 = real_biased[:50]
deg_top50  = deg_biased[:50]
real_top100 = real_biased[:100]
deg_top100  = deg_biased[:100]

def genes_to_idxs(genes):
    return [gene_to_idx[g] for g in genes if g in gene_to_idx]

real_only_idx = genes_to_idxs(real_only)
deg_only_idx  = genes_to_idxs(deg_only)
shared_idx    = genes_to_idxs(shared)

real_top50_idx = [x["idx"] for x in real_top50]
deg_top50_idx  = [x["idx"] for x in deg_top50]
real_top100_idx = [x["idx"] for x in real_top100]
deg_top100_idx  = [x["idx"] for x in deg_top100]

print(f"A shape = {A.shape}")
print(f"edges   = {A.nnz // 2}")
print(f"degree mean={deg.mean():.3f}, median={np.median(deg):.3f}, max={deg.max():.1f}")
print("-" * 80)

summarize_group("all genes", np.arange(len(symbols)))
summarize_group("shared (freq>=5 in both)", shared_idx)
summarize_group("real_only (freq>=5)", real_only_idx)
summarize_group("deg_only (freq>=5)", deg_only_idx)
summarize_group("real_biased_top50", real_top50_idx)
summarize_group("deg_biased_top50", deg_top50_idx)
summarize_group("real_biased_top100", real_top100_idx)
summarize_group("deg_biased_top100", deg_top100_idx)

print("-" * 80)
print("[top 20 real-biased genes with degree]")
for x in real_top50[:20]:
    print(f'{x["gene"]}\tdeg={deg[x["idx"]]:.0f}\treal={x["real_freq"]}\tdegnull={x["deg_freq"]}\tdiff={x["diff"]}')

print("-" * 80)
print("[top 20 degnull-biased genes with degree]")
for x in deg_top50[:20]:
    print(f'{x["gene"]}\tdeg={deg[x["idx"]]:.0f}\treal={x["real_freq"]}\tdegnull={x["deg_freq"]}\tdiff={x["diff"]}')