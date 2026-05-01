import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to Data_Cortex_Nuclear.csv")
    p.add_argument("--out_dir", type=str, required=True, help="Output dir under datasets/")
    p.add_argument("--task", type=str, default="genotype", choices=["genotype"])
    p.add_argument("--impute", type=str, default="median", choices=["median", "mean"])
    return p.parse_args()

def robust_read_table(path):
    import pandas as pd

    # 先读文件头判断类型
    with open(path, "rb") as f:
        head = f.read(8)

    # Excel 97-2003 .xls: D0 CF 11 E0 ...
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        print("[prepare] Detected Excel .xls (OLE). Using read_excel...")
        return pd.read_excel(path, engine="xlrd")

    # .xlsx/.xlsm: PK...
    if head.startswith(b"PK"):
        print("[prepare] Detected Excel .xlsx (zip). Using read_excel...")
        return pd.read_excel(path, engine="openpyxl")

    # 否则当作文本 CSV（保留你之前的兜底逻辑）
    candidates = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),
        ("utf-16", "strict"),
        ("cp1252", "strict"),
        ("latin1", "strict"),
        ("utf-8", "replace"),
        ("utf-16", "replace"),
        ("latin1", "replace"),
    ]
    last = None
    for enc, errmode in candidates:
        try:
            with open(path, "r", encoding=enc, errors=errmode, newline="") as f:
                try:
                    df = pd.read_csv(f, engine="python", on_bad_lines="skip")
                except TypeError:
                    df = pd.read_csv(f, engine="python", error_bad_lines=False, warn_bad_lines=True)
            print(f"[prepare] Loaded CSV with encoding={enc}, errors={errmode}, rows={len(df)}, cols={len(df.columns)}")
            return df
        except Exception as e:
            last = e

    raise RuntimeError(f"Failed to read file as excel/csv. Last error: {last}")


def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = robust_read_table(csv_path)

    # Mice Protein: first column is MouseID, last column is class (often named "class")
    # proteins columns are in the middle
    # robustly detect feature columns: float-like columns
    feature_cols = [c for c in df.columns if c not in ["MouseID", "Genotype", "Treatment", "Behavior", "class"]]
    # keep only numeric columns
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    # auto-detect label column
    label_candidates = ["class", "Class", "group", "Group"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column. Columns are: {list(df.columns)[:30]} ...")

    # label: genotype binary (c vs t) from class prefix
    cls = df[label_col].astype(str).values
    y = np.array([1 if s.startswith("t") else 0 for s in cls], dtype=np.int64)

    X = df[feature_cols].to_numpy(dtype=np.float32)

    # impute missing
    if np.isnan(X).any():
        if args.impute == "median":
            fill = np.nanmedian(X, axis=0)
        else:
            fill = np.nanmean(X, axis=0)
        # if a column is all-NaN, nanmedian gives NaN; drop those columns
        bad = np.isnan(fill)
        if bad.any():
            keep = ~bad
            X = X[:, keep]
            feature_cols = [c for c, k in zip(feature_cols, keep) if k]
            fill = fill[keep]
        inds = np.where(np.isnan(X))
        X[inds] = np.take(fill, inds[1])

    # save
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    meta = {
        "task": args.task,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_label": "t-* (trisomy)",
        "negative_label": "c-* (control)",
        "feature_names": feature_cols,
        "csv": str(csv_path),
        "impute": args.impute,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[prepare] Saved:", out_dir)
    print("[prepare] X:", X.shape, " y:", y.shape, " pos_rate:", float(y.mean()))


if __name__ == "__main__":
    main()
