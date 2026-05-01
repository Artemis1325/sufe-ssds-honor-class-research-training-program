from pathlib import Path
import subprocess
import sys

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[1]

SEEDS = list(range(42, 62))  # 42..61
RESULTS_DIR = ROOT / "results"
GAMMA = "0.001"

def run(cmd):
    print(">>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def train_one(data_dir: Path, run_name: str, seed: int):
    run([
        PYTHON,
        str(ROOT / "scripts" / "run_train.py"),
        "--data_dir", str(data_dir),
        "--results_dir", str(RESULTS_DIR),
        "--run_name", run_name,
        "--gamma", GAMMA,
        "--seed", str(seed),
    ])

def main():
    # real graph
    for s in SEEDS:
        train_one(
            ROOT / "datasets" / "tcga_brca",
            f"brca_real_s{s}_g1e3",
            s,
        )

    # ordinary random
    for s in SEEDS:
        train_one(
            ROOT / "datasets" / f"tcga_brca_random_s{s}",
            f"brca_random_s{s}_g1e3",
            s,
        )

    # degree-preserving null
    for s in SEEDS:
        train_one(
            ROOT / "datasets" / f"tcga_brca_degnull_s{s}",
            f"brca_degnull_s{s}_g1e3",
            s,
        )

    print("All training runs completed.")

if __name__ == "__main__":
    main()