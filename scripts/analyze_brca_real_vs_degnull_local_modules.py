import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "tcga_brca")
CMP_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")

EDGE_FILE = os.path.join(DATA_DIR, "reactome_edges_symbol_brca.tsv")
REAL_BIAS_FILE = os.path.join(CMP_DIR, "real_biased_diff_ge_5.tsv")
DEG_BIAS_FILE = os.path.join(CMP_DIR, "degnull_biased_diff_ge_5.tsv")

TOPN = 200  # 先看最强偏向的前200个基因

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
        # 尝试自动找symbol列
        lower = [x.lower() for x in header]
        # 常见情况：source target / gene1 gene2 / src dst
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
            # 如果没识别到，就默认前两列
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

real_rows = read_bias_table(REAL_BIAS_FILE)[:TOPN]
deg_rows = read_bias_table(DEG_BIAS_FILE)[:TOPN]

real_genes = [x["gene"] for x in real_rows]
deg_genes = [x["gene"] for x in deg_rows]

edges = load_edges(EDGE_FILE)

nbrs = defaultdict(set)
for u, v in edges:
    nbrs[u].add(v)
    nbrs[v].add(u)

all_graph_genes = set(nbrs.keys())

real_set = set(real_genes)
deg_set = set(deg_genes)

def summarize_group(name, genes):
    genes = [g for g in genes if g in all_graph_genes]
    internal_edge_count = 0
    partner_counter = Counter()
    degree_list = []

    seen = set()
    for g in genes:
        ng = nbrs[g]
        degree_list.append(len(ng))
        for h in ng:
            partner_counter[h] += 1
            if h in genes:
                a, b = sorted((g, h))
                seen.add((a, b))
    internal_edge_count = len(seen)

    print(f"[{name}]")
    print(f"genes in graph: {len(genes)}")
    if degree_list:
        degree_sorted = sorted(degree_list)
        mid = degree_sorted[len(degree_sorted)//2]
        print(f"mean degree in Reactome subgraph: {sum(degree_list)/len(degree_list):.3f}")
        print(f"median degree in Reactome subgraph: {mid}")
    print(f"internal edges among these genes: {internal_edge_count}")

    # 看它们最常共同连向哪些基因
    print(f"[top 30 shared neighbor / local hub signals for {name}]")
    for i, (gene, cnt) in enumerate(partner_counter.most_common(30), 1):
        tag = []
        if gene in real_set:
            tag.append("REAL")
        if gene in deg_set:
            tag.append("DEG")
        tag_str = ",".join(tag) if tag else "-"
        print(f"{i:02d}. {gene}\tcount={cnt}\ttag={tag_str}")
    print("-" * 80)

summarize_group("real_biased_top200", real_genes)
summarize_group("degnull_biased_top200", deg_genes)

# 再看两组基因的直接重叠和互连情况
real_in_graph = [g for g in real_genes if g in all_graph_genes]
deg_in_graph = [g for g in deg_genes if g in all_graph_genes]

overlap = sorted(set(real_in_graph) & set(deg_in_graph))
cross_edges = 0
for g in real_in_graph:
    for h in nbrs[g]:
        if h in set(deg_in_graph):
            cross_edges += 1

print("[cross-group summary]")
print(f"overlap genes between top{TOPN} real-biased and top{TOPN} deg-biased: {len(overlap)}")
print(f"directed cross edges real->deg in Reactome graph: {cross_edges}")
print("top overlap genes:", overlap[:30])