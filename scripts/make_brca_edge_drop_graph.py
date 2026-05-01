from pathlib import Path
import argparse
import shutil
import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "datasets" / "tcga_brca"


def copy_non_graph_files(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.name in {"A.npz", "L.npz"}:
            continue
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)


def drop_edges_from_adj(A: sp.csr_matrix, drop_rate: float, seed: int) -> sp.csr_matrix:
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    A = A.tocsr()
    if (A != A.T).nnz != 0:
        raise ValueError("A must be symmetric.")

    rng = np.random.default_rng(seed)

    # only keep upper triangle to represent undirected edges once
    U = sp.triu(A, k=1).tocoo()
    num_edges = U.nnz
    if num_edges == 0:
        raise ValueError("Graph has no edges.")

    keep_mask = rng.random(num_edges) > drop_rate

    row = U.row[keep_mask]
    col = U.col[keep_mask]
    data = np.ones_like(row, dtype=np.float32)

    n = A.shape[0]
    U_keep = sp.coo_matrix((data, (row, col)), shape=(n, n))
    A_keep = (U_keep + U_keep.T).tocsr()
    A_keep.eliminate_zeros()

    return A_keep


def laplacian_from_adj(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - A
    return L.tocsr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_rate", type=float, required=True, help="Fraction of edges to drop, e.g. 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for graph perturbation")
    parser.add_argument("--src_dir", type=str, default=str(SRC_DIR))
    parser.add_argument("--dst_dir", type=str, default=None)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    if args.dst_dir is None:
        pct = int(round(args.drop_rate * 100))
        dst_dir = src_dir.parent / f"tcga_brca_drop{pct}_s{args.seed}"
    else:
        dst_dir = Path(args.dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    A_path = src_dir / "A.npz"
    if not A_path.exists():
        raise FileNotFoundError(f"Missing adjacency file: {A_path}")

    A = sp.load_npz(A_path).tocsr()
    orig_edges = sp.triu(A, k=1).nnz

    A_drop = drop_edges_from_adj(A, args.drop_rate, args.seed)
    new_edges = sp.triu(A_drop, k=1).nnz
    L_drop = laplacian_from_adj(A_drop)

    copy_non_graph_files(src_dir, dst_dir)
    sp.save_npz(dst_dir / "A.npz", A_drop)
    sp.save_npz(dst_dir / "L.npz", L_drop)

    print(f"Saved perturbed dataset to: {dst_dir}")
    print(f"num_nodes = {A.shape[0]}")
    print(f"orig_edges = {orig_edges}")
    print(f"new_edges = {new_edges}")
    print(f"actual_drop_rate = {(orig_edges - new_edges) / orig_edges:.6f}")


if __name__ == "__main__":
    main()