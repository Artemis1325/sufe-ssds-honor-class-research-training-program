from pathlib import Path
import pandas as pd

base = Path(r"D:\LapReg_lassonet_project\datasets\tcga_brca")

expr_path = base / "TCGA-BRCA.star_tpm.tsv"
clinical_path = base / "TCGA-BRCA.clinical.tsv"

# 读 expression
expr = pd.read_csv(expr_path, sep="\t")
expr_samples = expr.columns[1:].tolist()

print("==== 1. expression 样本基本情况 ====")
print("n_expr_samples =", len(expr_samples))
print("expr sample preview:", expr_samples[:10])

# 读 clinical
clinical = pd.read_csv(clinical_path, sep="\t")

print("\n==== 2. clinical 列名 ====")
for c in clinical.columns:
    print(c)

print("\n==== 3. sample_type 相关列候选 ====")
cands = [c for c in clinical.columns if "sample" in c.lower() or "type" in c.lower() or "tumor" in c.lower() or "normal" in c.lower()]
print(cands)

print("\n==== 4. 每个候选列的前10个非空唯一值 ====")
for c in cands:
    vals = clinical[c].dropna().astype(str).unique().tolist()[:10]
    print(f"\n[{c}]")
    print(vals)

# 我们已知 sample 列存在
sample_col = "sample"
if sample_col not in clinical.columns:
    raise ValueError("clinical 里没有 sample 列，请停下来告诉我。")

print("\n==== 5. clinical sample 基本情况 ====")
clinical_samples = clinical[sample_col].astype(str).tolist()
print("n_clinical_samples =", len(clinical_samples))
print("clinical sample preview:", clinical_samples[:10])

expr_set = set(expr_samples)
clinical_set = set(clinical_samples)

overlap = sorted(expr_set & clinical_set)

print("\n==== 6. expression / clinical 样本交集 ====")
print("overlap =", len(overlap))
print("only_expr =", len(expr_set - clinical_set))
print("only_clinical =", len(clinical_set - expr_set))
print("overlap preview:", overlap[:10])

print("\n==== 7. 尝试用 TCGA barcode 第4段判断样本类型 ====")
def infer_sample_type(barcode: str):
    parts = barcode.split("-")
    if len(parts) < 4:
        return "UNKNOWN"
    code = parts[3][:2]
    if code.isdigit():
        code = int(code)
        if 1 <= code <= 9:
            return "Primary Tumor"
        if 10 <= code <= 19:
            return "Solid Tissue Normal"
    return "OTHER"

tmp = pd.DataFrame({"sample": overlap})
tmp["barcode_type"] = tmp["sample"].map(infer_sample_type)

print(tmp["barcode_type"].value_counts(dropna=False))

print("\n==== 8. overlap 样本中 tumor/normal 预览 ====")
print(tmp.head(20))