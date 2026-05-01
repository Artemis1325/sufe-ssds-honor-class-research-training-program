from pathlib import Path
import subprocess
import sys

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[1]

REAL_DIR = ROOT / "datasets" / "tcga_brca"
X_SRC = REAL_DIR / "X.npy"

SEEDS = list(range(42, 62))  # 42..61

def run(cmd):
    print(">>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def ensure_x(dst_dir: Path):
    dst = dst_dir / "X.npy"
    if not dst.exists():
        dst.write_bytes(X_SRC.read_bytes())

def main():
    # ordinary random
    for s in SEEDS:
        out_dir = ROOT / "datasets" / f"tcga_brca_random_s{s}"
        if not out_dir.exists():
            run([
                PYTHON,
                str(ROOT / "scripts" / "make_brca_random_graph.py"),
                "--input_dir", str(REAL_DIR),
                "--output_dir", str(out_dir),
                "--seed", str(s),
            ])
        ensure_x(out_dir)

    # degree-preserving null
    for s in SEEDS:
        out_dir = ROOT / "datasets" / f"tcga_brca_degnull_s{s}"
        if not out_dir.exists():
            run([
                PYTHON,
                str(ROOT / "scripts" / "make_brca_degree_preserving_graph.py"),
                "--input_dir", str(REAL_DIR),
                "--output_dir", str(out_dir),
                "--seed", str(s),
                "--swap_factor", "10",
            ])
        ensure_x(out_dir)

    print("All graph priors generated.")

if __name__ == "__main__":
    main()