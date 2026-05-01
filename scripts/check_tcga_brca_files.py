from pathlib import Path
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

files = {
    "expr": base / "TCGA-BRCA.star_tpm.tsv",
    "clinical": base / "TCGA-BRCA.clinical.tsv",
    "reactome": base / "reactome.homo_sapiens.interactions.tab-delimited.txt",
    "gencode": base / "gencode.v49.basic.annotation.gtf",
}

print("==== 1. 文件是否存在 ====")
for k, p in files.items():
    print(f"{k}: {p.exists()} -> {p}")

print("\n==== 2. expression 前5行前5列 ====")
expr = pd.read_csv(files["expr"], sep="\t", nrows=5)
print(expr.iloc[:, :5])
print("shape (preview):", expr.shape)
print("columns preview:", expr.columns[:5].tolist())

print("\n==== 3. clinical 前5行前8列 ====")
clinical = pd.read_csv(files["clinical"], sep="\t", nrows=5)
print(clinical.iloc[:, :8])
print("shape (preview):", clinical.shape)
print("columns preview:", clinical.columns[:12].tolist())

print("\n==== 4. reactome 前5行 ====")
with open(files["reactome"], "r", encoding="utf-8") as f:
    for i in range(5):
        print(f.readline().rstrip())

print("\n==== 5. gencode 前10行 ====")
with open(files["gencode"], "r", encoding="utf-8") as f:
    for i in range(10):
        print(f.readline().rstrip())