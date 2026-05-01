from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from lapreg_lassonet.models.graph_lassonet import GraphLassoNet


def find_single_file(run_dir: Path, pattern: str):
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {run_dir}")
    if len(files) > 1:
        print(f"[warn] multiple files matched {pattern}, using latest: {files[-1].name}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--external_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    external_dir = Path(args.external_dir)
    device = torch.device(args.device)

    model_fp = find_single_file(run_dir, "model_*.pt")
    scaler_fp = find_single_file(run_dir, "scaler_*.npz")

    ckpt = torch.load(model_fp, map_location=device)
    scaler = np.load(scaler_fp)

    X = np.load(external_dir / "X_graph.npy").astype(np.float32)
    y = np.load(external_dir / "y.npy").astype(np.int64)

    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    if X.shape[1] != mean.shape[0]:
        raise ValueError(f"Feature mismatch: X has {X.shape[1]} cols, scaler has {mean.shape[0]} dims")

    Xz = (X - mean) / std

    hidden_dims = ckpt["hidden_dims"]
    if isinstance(hidden_dims, list):
        hidden_dims = tuple(hidden_dims)

    model = GraphLassoNet(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(Xz).to(device)
        logits = model(xb).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        print("prob min =", float(probs.min()))
        print("prob max =", float(probs.max()))
        print("prob mean =", float(probs.mean()))
        print("prob std =", float(probs.std()))
        print("first 10 probs =", probs[:10].tolist())

    pred = (probs >= 0.5).astype(np.int64)

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, pred)

    out = {
        "run_dir": str(run_dir),
        "external_dir": str(external_dir),
        "n_samples": int(len(y)),
        "tumor_count": int((y == 1).sum()),
        "normal_count": int((y == 0).sum()),
        "acc": float(acc),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "model_file": str(model_fp),
        "scaler_file": str(scaler_fp),
    }

    out_fp = run_dir / "external_test_gse15852.json"
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", out_fp)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()