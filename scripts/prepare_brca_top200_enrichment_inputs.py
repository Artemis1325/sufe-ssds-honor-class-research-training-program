import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")
OUT_DIR = os.path.join(CMP_DIR, "enrichment_inputs")

os.makedirs(OUT_DIR, exist_ok=True)

def read_top_genes_from_bias_tsv(fp, topn=200):
    genes = []
    with open(fp, "r", encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if i >= topn:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            gene = parts[2].strip()
            if gene:
                genes.append(gene)
    return genes

def write_genes(genes, fp):
    genes = [g for g in genes if g]
    with open(fp, "w", encoding="utf-8") as f:
        for g in genes:
            f.write(g + "\n")

real_top200 = read_top_genes_from_bias_tsv(
    os.path.join(CMP_DIR, "real_biased_top200.tsv"), topn=200
)
deg_top200 = read_top_genes_from_bias_tsv(
    os.path.join(CMP_DIR, "degnull_biased_top200.tsv"), topn=200
)

real_fp = os.path.join(OUT_DIR, "real_biased_top200.txt")
deg_fp = os.path.join(OUT_DIR, "degnull_biased_top200.txt")

write_genes(real_top200, real_fp)
write_genes(deg_top200, deg_fp)

print("[saved]")
print(real_fp, len(real_top200))
print(deg_fp, len(deg_top200))

print("-" * 80)
print("[real top 20 preview]")
for i, g in enumerate(real_top200[:20], 1):
    print(f"{i:02d}. {g}")

print("-" * 80)
print("[degnull top 20 preview]")
for i, g in enumerate(deg_top200[:20], 1):
    print(f"{i:02d}. {g}")