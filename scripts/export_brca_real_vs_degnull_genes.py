import os
import glob
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "tcga_brca")
OUT_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")

TOPK = 50
MIN_FREQ = 5
SEEDS = list(range(42, 62))  # 42..61

symbol_file = os.path.join(DATA_DIR, "feature_graph_symbols.txt")
with open(symbol_file, "r", encoding="utf-8") as f:
    symbols = [x.strip() for x in f if x.strip()]

assert len(symbols) == 5203, f"feature_graph_symbols length != 5203, got {len(symbols)}"

os.makedirs(OUT_DIR, exist_ok=True)

def find_selected_file(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, "selected_*.npy")))
    if not files:
        raise FileNotFoundError(f"No selected_*.npy found in {run_dir}")
    return files[-1]  # 取最新一个，避免旧结果残留

def load_mask(fp: str):
    arr = np.load(fp, allow_pickle=True)
    if arr.shape != (5203,):
        raise ValueError(f"Unexpected shape for {fp}: {arr.shape}")
    arr = np.asarray(arr)
    uniq = np.unique(arr)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"{fp} is not a 0/1 mask. unique values={uniq[:20]}")
    return arr.astype(int)

def collect_group(prefix: str):
    counter = Counter()
    per_seed_nnz = []
    used_files = []

    for seed in SEEDS:
        run_dir = os.path.join(ROOT, "results", f"{prefix}_s{seed}_g1e3")
        fp = find_selected_file(run_dir)
        mask = load_mask(fp)
        idx = np.flatnonzero(mask)
        counter.update(idx.tolist())
        per_seed_nnz.append((seed, int(mask.sum())))
        used_files.append(fp)

    return counter, per_seed_nnz, used_files

real_counter, real_nnz, real_files = collect_group("brca_real")
deg_counter, deg_nnz, deg_files = collect_group("brca_degnull")

def counter_to_rows(counter):
    rows = []
    for idx, freq in counter.most_common():
        rows.append((idx, symbols[idx], freq))
    return rows

real_rows = counter_to_rows(real_counter)
deg_rows = counter_to_rows(deg_counter)

real_top = real_rows[:TOPK]
deg_top = deg_rows[:TOPK]

real_set = {gene for _, gene, freq in real_rows if freq >= MIN_FREQ}
deg_set = {gene for _, gene, freq in deg_rows if freq >= MIN_FREQ}

shared = sorted(real_set & deg_set)
real_only = sorted(real_set - deg_set)
deg_only = sorted(deg_set - real_set)

def write_table(rows, fp):
    with open(fp, "w", encoding="utf-8") as f:
        f.write("rank\tindex\tgene\tfreq\n")
        for i, (idx, gene, freq) in enumerate(rows, 1):
            f.write(f"{i}\t{idx}\t{gene}\t{freq}\n")

def write_list(items, fp):
    with open(fp, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")

def write_seed_nnz(rows, fp):
    with open(fp, "w", encoding="utf-8") as f:
        f.write("seed\tnnz\n")
        for seed, nnz in rows:
            f.write(f"{seed}\t{nnz}\n")

write_table(real_top, os.path.join(OUT_DIR, "real_top50.tsv"))
write_table(deg_top, os.path.join(OUT_DIR, "degnull_top50.tsv"))
write_list(shared, os.path.join(OUT_DIR, "shared_minfreq5.txt"))
write_list(real_only, os.path.join(OUT_DIR, "real_only_minfreq5.txt"))
write_list(deg_only, os.path.join(OUT_DIR, "degnull_only_minfreq5.txt"))
write_seed_nnz(real_nnz, os.path.join(OUT_DIR, "real_seed_nnz.tsv"))
write_seed_nnz(deg_nnz, os.path.join(OUT_DIR, "degnull_seed_nnz.tsv"))

print("[real files used]")
for x in real_files:
    print(x)

print("-" * 80)
print("[degnull files used]")
for x in deg_files:
    print(x)

print("-" * 80)
print(f"real_set (freq>={MIN_FREQ}): {len(real_set)}")
print(f"deg_set  (freq>={MIN_FREQ}): {len(deg_set)}")
print(f"shared: {len(shared)}")
print(f"real_only: {len(real_only)}")
print(f"deg_only: {len(deg_only)}")

print("-" * 80)
print("[real top 20]")
for i, (idx, gene, freq) in enumerate(real_top[:20], 1):
    print(f"{i:02d}. {gene}\t(freq={freq}, idx={idx})")

print("-" * 80)
print("[degnull top 20]")
for i, (idx, gene, freq) in enumerate(deg_top[:20], 1):
    print(f"{i:02d}. {gene}\t(freq={freq}, idx={idx})")

print("-" * 80)
print(f"Saved to: {OUT_DIR}")