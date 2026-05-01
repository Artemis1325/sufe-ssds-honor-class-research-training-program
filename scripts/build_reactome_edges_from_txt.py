from pathlib import Path
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

reactome_path = base / "reactome.homo_sapiens.interactions.tab-delimited.txt"

rows = []

with open(reactome_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 7:
            continue

        rows.append({
            "id1_raw": parts[1],   # Interactor 1 Ensembl gene id
            "id2_raw": parts[4],   # Interactor 2 Ensembl gene id
            "edge_type": parts[6],
        })

df = pd.DataFrame(rows)

print("==== 1. raw parsed rows ====")
print("shape =", df.shape)
print(df.head())

def extract_gene_ids(s: str):
    """
    从这种字符串里提取 ENSG:
    ENSEMBL:ENST...|ENSEMBL:ENSP...|ENSEMBL:ENSG00000136156
    只保留 ENSGxxxx
    """
    if pd.isna(s):
        return []
    out = []
    for item in str(s).split("|"):
        item = item.strip()
        if item.startswith("ENSEMBL:ENSG"):
            out.append(item.replace("ENSEMBL:", ""))
    return out

edge_rows = []

for _, row in df.iterrows():
    genes1 = extract_gene_ids(row["id1_raw"])
    genes2 = extract_gene_ids(row["id2_raw"])
    etype = row["edge_type"]

    if not genes1 or not genes2:
        continue

    for g1 in genes1:
        for g2 in genes2:
            if g1 == g2:
                continue
            a, b = sorted([g1, g2])
            edge_rows.append((a, b, etype))

edge_df = pd.DataFrame(edge_rows, columns=["ensembl_a", "ensembl_b", "edge_type"])
edge_df = edge_df.drop_duplicates().reset_index(drop=True)

print("\n==== 2. edge table after ENSG extraction ====")
print("shape =", edge_df.shape)
print(edge_df.head(10))

print("\n==== 3. edge_type top 20 ====")
print(edge_df["edge_type"].value_counts().head(20))

out_path = base / "reactome_edges_ensembl.tsv"
edge_df.to_csv(out_path, sep="\t", index=False)

print("\nsaved:", out_path)