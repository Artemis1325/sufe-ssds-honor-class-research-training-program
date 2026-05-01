import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "tcga_brca")
CMP_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")
OUT_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")

EDGE_FILE = os.path.join(DATA_DIR, "reactome_edges_symbol_brca.tsv")
REAL_BIAS_FILE = os.path.join(CMP_DIR, "real_biased_diff_ge_5.tsv")
DEG_BIAS_FILE = os.path.join(CMP_DIR, "degnull_biased_diff_ge_5.tsv")

TOPN = 200
TOP_HUBS = 15

def read_bias_table(fp):
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

def load_edges(fp):
    edges = []
    with open(fp, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        lower = [x.lower() for x in header]

        cand_pairs = [
            ("source", "target"),
            ("src", "dst"),
            ("gene1", "gene2"),
            ("symbol1", "symbol2"),
            ("from", "to"),
        ]
        c1 = c2 = None
        for a, b in cand_pairs:
            if a in lower and b in lower:
                c1, c2 = lower.index(a), lower.index(b)
                break
        if c1 is None:
            c1, c2 = 0, 1

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(c1, c2):
                continue
            u = parts[c1].strip()
            v = parts[c2].strip()
            if u and v and u != v:
                edges.append((u, v))
    return edges

def analyze_group(name, genes, nbrs):
    gene_set = set(genes)
    genes_in_graph = [g for g in genes if g in nbrs]
    gene_set_in_graph = set(genes_in_graph)

    # internal edges
    seen = set()
    partner_counter = Counter()
    degree_list = []

    for g in genes_in_graph:
        ng = nbrs[g]
        degree_list.append(len(ng))
        for h in ng:
            partner_counter[h] += 1
            if h in gene_set_in_graph:
                a, b = sorted((g, h))
                seen.add((a, b))

    internal_edges = len(seen)
    n = len(genes_in_graph)
    possible_edges = n * (n - 1) // 2
    density = internal_edges / possible_edges if possible_edges > 0 else 0.0

    top_hubs = partner_counter.most_common(TOP_HUBS)

    return {
        "name": name,
        "n": n,
        "internal_edges": internal_edges,
        "density": density,
        "mean_degree": sum(degree_list) / len(degree_list) if degree_list else 0.0,
        "median_degree": sorted(degree_list)[len(degree_list)//2] if degree_list else 0,
        "top_hubs": top_hubs,
    }

real_rows = read_bias_table(REAL_BIAS_FILE)[:TOPN]
deg_rows = read_bias_table(DEG_BIAS_FILE)[:TOPN]
real_genes = [x["gene"] for x in real_rows]
deg_genes = [x["gene"] for x in deg_rows]

edges = load_edges(EDGE_FILE)
nbrs = defaultdict(set)
for u, v in edges:
    nbrs[u].add(v)
    nbrs[v].add(u)

real_res = analyze_group("real_biased_top200", real_genes, nbrs)
deg_res = analyze_group("degnull_biased_top200", deg_genes, nbrs)

os.makedirs(OUT_DIR, exist_ok=True)

summary_fp = os.path.join(OUT_DIR, "module_summary_top200.tsv")
with open(summary_fp, "w", encoding="utf-8") as f:
    f.write("group\tn_genes\tinternal_edges\tedge_density\tmean_degree\tmedian_degree\n")
    for res in [real_res, deg_res]:
        f.write(
            f'{res["name"]}\t{res["n"]}\t{res["internal_edges"]}\t'
            f'{res["density"]:.6f}\t{res["mean_degree"]:.3f}\t{res["median_degree"]}\n'
        )

hub_fp = os.path.join(OUT_DIR, "module_top_hubs_top200.tsv")
with open(hub_fp, "w", encoding="utf-8") as f:
    f.write("group\trank\thub_gene\tneighbor_count_within_biased_set\n")
    for res in [real_res, deg_res]:
        for i, (gene, cnt) in enumerate(res["top_hubs"], 1):
            f.write(f'{res["name"]}\t{i}\t{gene}\t{cnt}\n')

print("[module summary]")
for res in [real_res, deg_res]:
    print(
        f'{res["name"]}: '
        f'n={res["n"]}, internal_edges={res["internal_edges"]}, '
        f'density={res["density"]:.6f}, mean_degree={res["mean_degree"]:.3f}, '
        f'median_degree={res["median_degree"]}'
    )

print("-" * 80)
print("[top hubs: real_biased_top200]")
for i, (gene, cnt) in enumerate(real_res["top_hubs"], 1):
    print(f"{i:02d}. {gene}\t{cnt}")

print("-" * 80)
print("[top hubs: degnull_biased_top200]")
for i, (gene, cnt) in enumerate(deg_res["top_hubs"], 1):
    print(f"{i:02d}. {gene}\t{cnt}")

print("-" * 80)
print("saved:")
print(summary_fp)
print(hub_fp)