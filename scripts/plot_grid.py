import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_json", type=str, required=True,
                   help="Path or glob pattern to grid json (e.g. results/grid_sanity/grid_*.json)")
    p.add_argument("--metric", type=str, default="acc",
                   choices=["acc", "auc", "nnz_theta", "lap_energy"])
    return p.parse_args()


def main():
    args = parse_args()

    # ---- NEW: glob support ----
    paths = glob.glob(args.grid_json)
    if len(paths) == 0:
        raise FileNotFoundError(f"No files match pattern: {args.grid_json}")
    if len(paths) > 1:
        print(f"[plot_grid] Multiple files found, using latest:")
        for p in paths:
            print("  ", p)
        paths = sorted(paths)
    grid_path = paths[-1]

    print("[plot_grid] Using:", grid_path)

    with open(grid_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    lambdas = obj["lambdas"]
    gammas = obj["gammas"]
    results = obj["results"]

    # build matrix [len(lambdas), len(gammas)]
    M = np.full((len(lambdas), len(gammas)), np.nan, dtype=float)
    for r in results:
        i = lambdas.index(r["lambda_l1"])
        j = gammas.index(r["gamma"])
        M[i, j] = float(r["final"][args.metric])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto", origin="lower")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([str(g) for g in gammas], rotation=45, ha="right")
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels([str(l) for l in lambdas])
    ax.set_xlabel("gamma")
    ax.set_ylabel("lambda_l1")
    ax.set_title(f"Grid heatmap: {args.metric}")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
