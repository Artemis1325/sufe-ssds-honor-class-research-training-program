from pathlib import Path

g0_path = Path("results/brca_sanity_g0/selected_genes_seed42.txt")
g1_path = Path("results/brca_sanity_g0p001/selected_genes_seed42.txt")

g0 = {line.strip() for line in open(g0_path, "r", encoding="utf-8") if line.strip()}
g1 = {line.strip() for line in open(g1_path, "r", encoding="utf-8") if line.strip()}

inter = g0 & g1
union = g0 | g1
only_g0 = g0 - g1
only_g1 = g1 - g0

jaccard = len(inter) / len(union)

print("gamma=0 size:", len(g0))
print("gamma=0.001 size:", len(g1))
print("intersection:", len(inter))
print("union:", len(union))
print("jaccard:", jaccard)
print("only gamma=0:", len(only_g0))
print("only gamma=0.001:", len(only_g1))

print("\nfirst 20 only gamma=0 genes:")
for x in sorted(list(only_g0))[:20]:
    print(x)

print("\nfirst 20 only gamma=0.001 genes:")
for x in sorted(list(only_g1))[:20]:
    print(x)