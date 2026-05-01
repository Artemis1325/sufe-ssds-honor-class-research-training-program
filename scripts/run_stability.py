import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import glob
import os
import numpy as np

from lapreg_lassonet.eval.metrics import jaccard_similarity
from lapreg_lassonet.utils.io import save_json, results_subdir, timestamp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default="graph_lassonet")
    p.add_argument("--pattern", type=str, default="selected_*.npy")  # load saved selected masks
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = results_subdir(args.results_dir, args.run_name)

    files = sorted(glob.glob(os.path.join(run_dir, args.pattern)))
    if len(files) < 2:
        raise RuntimeError(f"Need >=2 selected masks in {run_dir}, got {len(files)}")

    masks = [np.load(f).astype(int) for f in files]

    # pairwise jaccard
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            vals.append(jaccard_similarity(masks[i], masks[j]))

    out = {
        "run_dir": run_dir,
        "num_masks": len(masks),
        "mean_jaccard": float(np.mean(vals)),
        "std_jaccard": float(np.std(vals)),
        "min_jaccard": float(np.min(vals)),
        "max_jaccard": float(np.max(vals)),
        "files": files,
    }

    ts = timestamp()
    save_path = os.path.join(run_dir, f"stability_{ts}.json")
    save_json(save_path, out)
    print("Saved:", save_path)
    print(out)


if __name__ == "__main__":
    main()
