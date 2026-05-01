import os
import glob
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "tcga_brca")
OUT_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")
SEEDS = list(range(42, 62))

symbol_file = os.path.join(DATA_DIR, "feature_graph_symbols.txt")
with open(symbol_file, "r", encoding="utf-8") as f:
    symbols = [x.strip() for x in f if x.strip()]

assert len(symbols) == 5203

def find_selected_file(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, "selected_*.npy")))
    if not files:
        raise FileNotFoundError(f"No selected_*.npy found in {run_dir}")
    return files[-1]

def load_mask(fp: str):
    arr = np.load(fp, allow_pickle=True)
    arr = np.asarray(arr).astype(int)
    if arr.shape != (5203,):
        raise ValueError(f"Unexpected shape for {fp}: {arr.shape}")
    return arr

def collect_group(prefix: str):
    counts = np.zeros(5203, dtype=int)
    for seed in SEEDS:
        run_dir = os.path.join(ROOT, "results", f"{prefix}_s{seed}_g1e3")
        fp = find_selected_file(run_dir)
        mask = load_mask(fp)
        counts += mask
    return counts

real_counts = collect_group("brca_real")
deg_counts = collect_group("brca_degnull")

rows = []
for i, gene in enumerate(symbols):
    r = int(real_counts[i])
    d = int(deg_counts[i])
    diff = r - d
    rows.append((i, gene, r, d, diff))

# real更常选中
real_biased = sorted(rows, key=lambda x: (-x[4], -x[2], x[1]))
# degnull更常选中
deg_biased = sorted(rows, key=lambda x: (x[4], -x[3], x[1]))

os.makedirs(OUT_DIR, exist_ok=True)

def write_rows(items, fp):
    with open(fp, "w", encoding="utf-8") as f:
        f.write("rank\tindex\tgene\treal_freq\tdegnull_freq\tdiff_real_minus_degnull\n")
        for k, (idx, gene, r, d, diff) in enumerate(items, 1):
            f.write(f"{k}\t{idx}\t{gene}\t{r}\t{d}\t{diff}\n")

write_rows(real_biased[:200], os.path.join(OUT_DIR, "real_biased_top200.tsv"))
write_rows(deg_biased[:200], os.path.join(OUT_DIR, "degnull_biased_top200.tsv"))

# 再导出更严格的候选：差值>=5
real_strong = [x for x in real_biased if x[4] >= 5]
deg_strong = [x for x in deg_biased if x[4] <= -5]

write_rows(real_strong, os.path.join(OUT_DIR, "real_biased_diff_ge_5.tsv"))
write_rows(deg_strong, os.path.join(OUT_DIR, "degnull_biased_diff_ge_5.tsv"))

print("[summary]")
print("genes with real_freq > degnull_freq :", sum(1 for x in rows if x[4] > 0))
print("genes with real_freq < degnull_freq :", sum(1 for x in rows if x[4] < 0))
print("genes with equal freq               :", sum(1 for x in rows if x[4] == 0))
print("real-biased strong candidates (diff>=5):", len(real_strong))
print("deg-biased strong candidates (diff<=-5):", len(deg_strong))

print("-" * 80)
print("[top 30 real-biased genes]")
for k, (idx, gene, r, d, diff) in enumerate(real_biased[:30], 1):
    print(f"{k:02d}. {gene}\treal={r}\tdeg={d}\tdiff={diff}\tidx={idx}")

print("-" * 80)
print("[top 30 degnull-biased genes]")
for k, (idx, gene, r, d, diff) in enumerate(deg_biased[:30], 1):
    print(f"{k:02d}. {gene}\treal={r}\tdeg={d}\tdiff={diff}\tidx={idx}")

print("-" * 80)
print(f"Saved to: {OUT_DIR}")