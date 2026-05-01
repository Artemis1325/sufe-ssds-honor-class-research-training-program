import os
import glob
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "tcga_brca")
RESULTS_DIR = os.path.join(ROOT, "results")

REAL_PREFIX = "brca_real_multiseed20"
DROP_PREFIXES = [
    "brca_drop10_s42_multiseed20",
    "brca_drop30_s42_multiseed20",
    "brca_drop50_s42_multiseed20",
]

TOPK = 50
MIN_FREQ = 5

symbol_file = os.path.join(DATA_DIR, "feature_graph_symbols.txt")
with open(symbol_file, "r", encoding="utf-8") as f:
    symbols = [x.strip() for x in f if x.strip()]

assert len(symbols) == 5203, f"feature_graph_symbols length != 5203, got {len(symbols)}"


def find_selected_files(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, "selected_*.npy")))
    if not files:
        raise FileNotFoundError(f"No selected_*.npy found in {run_dir}")
    return files


def load_mask(fp: str):
    arr = np.load(fp, allow_pickle=True)
    if arr.shape != (5203,):
        raise ValueError(f"Unexpected shape for {fp}: {arr.shape}")
    arr = np.asarray(arr)
    uniq = np.unique(arr)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"{fp} is not a 0/1 mask. unique values={uniq[:20]}")
    return arr.astype(int)


def collect_group(run_name: str):
    run_dir = os.path.join(RESULTS_DIR, run_name)
    files = find_selected_files(run_dir)

    counter = Counter()
    per_file_nnz = []
    used_files = []

    for fp in files:
        mask = load_mask(fp)
        idx = np.flatnonzero(mask)
        counter.update(idx.tolist())
        per_file_nnz.append((os.path.basename(fp), int(mask.sum())))
        used_files.append(fp)

    return counter, per_file_nnz, used_files


def counter_to_rows(counter):
    rows = []
    for idx, freq in counter.most_common():
        rows.append((idx, symbols[idx], freq))
    return rows


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
        f.write("file\tnnz\n")
        for name, nnz in rows:
            f.write(f"{name}\t{nnz}\n")


def write_bias_table(real_rows, drop_rows, out_fp, topn=None):
    real_map = {gene: (idx, freq) for idx, gene, freq in real_rows}
    drop_map = {gene: (idx, freq) for idx, gene, freq in drop_rows}
    all_genes = sorted(set(real_map) | set(drop_map))

    rows = []
    for gene in all_genes:
        idx = real_map.get(gene, drop_map.get(gene))[0]
        real_f = real_map.get(gene, (idx, 0))[1]
        drop_f = drop_map.get(gene, (idx, 0))[1]
        diff = real_f - drop_f
        rows.append((idx, gene, real_f, drop_f, diff))

    rows = sorted(rows, key=lambda x: (-abs(x[4]), -x[2], x[1]))
    rows = [x for x in rows if x[4] != 0]

    if topn is not None:
        rows = rows[:topn]

    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("rank\tindex\tgene\treal_freq\tdrop_freq\tdiff\n")
        for i, (idx, gene, real_f, drop_f, diff) in enumerate(rows, 1):
            f.write(f"{i}\t{idx}\t{gene}\t{real_f}\t{drop_f}\t{diff}\n")

real_counter, real_nnz, real_files = collect_group(REAL_PREFIX)
real_rows = counter_to_rows(real_counter)
real_top = real_rows[:TOPK]
real_set = {gene for _, gene, freq in real_rows if freq >= MIN_FREQ}

print("[real files used]")
for x in real_files:
    print(x)
print("-" * 80)

for drop_prefix in DROP_PREFIXES:
    out_dir = os.path.join(RESULTS_DIR, f"gene_compare_{REAL_PREFIX}_vs_{drop_prefix}")
    os.makedirs(out_dir, exist_ok=True)

    drop_counter, drop_nnz, drop_files = collect_group(drop_prefix)
    drop_rows = counter_to_rows(drop_counter)
    drop_top = drop_rows[:TOPK]
    drop_set = {gene for _, gene, freq in drop_rows if freq >= MIN_FREQ}

    shared = sorted(real_set & drop_set)
    real_only = sorted(real_set - drop_set)
    drop_only = sorted(drop_set - real_set)

    write_table(real_top, os.path.join(out_dir, "real_top50.tsv"))
    write_table(drop_top, os.path.join(out_dir, "drop_top50.tsv"))
    write_list(shared, os.path.join(out_dir, "shared_minfreq5.txt"))
    write_list(real_only, os.path.join(out_dir, "real_only_minfreq5.txt"))
    write_list(drop_only, os.path.join(out_dir, "drop_only_minfreq5.txt"))
    write_seed_nnz(real_nnz, os.path.join(out_dir, "real_file_nnz.tsv"))
    write_seed_nnz(drop_nnz, os.path.join(out_dir, "drop_file_nnz.tsv"))
    write_bias_table(real_rows, drop_rows, os.path.join(out_dir, "real_vs_drop_biased_top200.tsv"), topn=200)
    write_bias_table(real_rows, drop_rows, os.path.join(out_dir, "real_vs_drop_diff_all.tsv"), topn=None)

    print(f"[drop files used: {drop_prefix}]")
    for x in drop_files:
        print(x)

    print("-" * 80)
    print(f"[compare: real vs {drop_prefix}]")
    print(f"real_set (freq>={MIN_FREQ}): {len(real_set)}")
    print(f"drop_set (freq>={MIN_FREQ}): {len(drop_set)}")
    print(f"shared: {len(shared)}")
    print(f"real_only: {len(real_only)}")
    print(f"drop_only: {len(drop_only)}")

    print("-" * 80)
    print("[real top 20]")
    for i, (idx, gene, freq) in enumerate(real_top[:20], 1):
        print(f"{i:02d}. {gene}\t(freq={freq}, idx={idx})")

    print("-" * 80)
    print("[drop top 20]")
    for i, (idx, gene, freq) in enumerate(drop_top[:20], 1):
        print(f"{i:02d}. {gene}\t(freq={freq}, idx={idx})")

    print("-" * 80)
    print(f"Saved to: {out_dir}")
    print("=" * 100)