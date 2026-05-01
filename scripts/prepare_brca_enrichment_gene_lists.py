import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP_DIR = os.path.join(ROOT, "results", "brca_gene_compare_g1e3")
OUT_DIR = os.path.join(CMP_DIR, "enrichment_inputs")

os.makedirs(OUT_DIR, exist_ok=True)

def read_txt_genes(fp):
    with open(fp, "r", encoding="utf-8") as f:
        genes = [x.strip() for x in f if x.strip()]
    return genes

def read_bias_tsv(fp):
    genes = []
    with open(fp, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            gene = parts[2].strip()
            if gene:
                genes.append(gene)
    return genes

def write_genes(genes, fp):
    genes = [g for g in genes if g]
    genes = sorted(set(genes))
    with open(fp, "w", encoding="utf-8") as f:
        for g in genes:
            f.write(g + "\n")
    return genes

inputs = {
    "real_only_minfreq5": read_txt_genes(os.path.join(CMP_DIR, "real_only_minfreq5.txt")),
    "degnull_only_minfreq5": read_txt_genes(os.path.join(CMP_DIR, "degnull_only_minfreq5.txt")),
    "real_biased_diff_ge_5": read_bias_tsv(os.path.join(CMP_DIR, "real_biased_diff_ge_5.tsv")),
    "degnull_biased_diff_ge_5": read_bias_tsv(os.path.join(CMP_DIR, "degnull_biased_diff_ge_5.tsv")),
}

print("[gene list sizes]")
for name, genes in inputs.items():
    uniq = write_genes(genes, os.path.join(OUT_DIR, f"{name}.txt"))
    print(f"{name}: raw={len(genes)}, unique={len(uniq)}")

print("-" * 80)
print("[saved files]")
for name in inputs:
    print(os.path.join(OUT_DIR, f"{name}.txt"))