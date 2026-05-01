import os
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "tcga_brca"
RESULTS_DIR = ROOT / "results"
SEEDS = list(range(42, 62))  # 42..61


def run(cmd):
    print(">>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def train_one(seed: int, graph_mode: str, pearson_k: int):
    run_name = f"brca_{graph_mode}_s{seed}_k{pearson_k}" if graph_mode == "pearson_train" else f"brca_{graph_mode}_s{seed}"
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "run_train.py"),
        "--data_dir", str(DATA_DIR),
        "--results_dir", str(RESULTS_DIR),
        "--run_name", run_name,
        "--gamma", "0.001",
        "--seed", str(seed),
        "--graph_mode", graph_mode,
    ]
    if graph_mode == "pearson_train":
        cmd.extend(["--pearson_k", str(pearson_k)])
    run(cmd)


def main():
    pearson_k = int(os.environ.get("BRCA_PEARSON_K", "10"))

    for seed in SEEDS:
        train_one(seed=seed, graph_mode="fixed", pearson_k=pearson_k)

    for seed in SEEDS:
        train_one(seed=seed, graph_mode="pearson_train", pearson_k=pearson_k)

    print("BRCA graph mode comparison completed.")


if __name__ == "__main__":
    main()
