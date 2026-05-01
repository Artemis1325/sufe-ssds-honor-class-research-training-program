import argparse
from pathlib import Path
import shutil

import numpy as np
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets/tcga_brca")
    parser.add_argument("--output_dir", type=str, default="datasets/tcga_brca_random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # 读取原图
    A = sp.load_npz(src / "A.npz").tocsr()
    n = A.shape[0]

    # 无向边数（按上三角算）
    A_triu = sp.triu(A, k=1).tocsr()
    m = A_triu.nnz
    print(f"n={n}, undirected edges={m}")

    # 随机采样 m 条无向边，禁止自环、禁止重复
    chosen = set()
    while len(chosen) < m:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        chosen.add((i, j))

    rows = []
    cols = []
    for i, j in chosen:
        rows.extend([i, j])
        cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.float32)
    A_rand = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A_rand.sum_duplicates()
    A_rand.setdiag(0)
    A_rand.eliminate_zeros()

    # 构造 L = D - A
    deg = np.asarray(A_rand.sum(axis=1)).ravel()
    L_rand = sp.diags(deg) - A_rand

    # 保存，覆盖新目录中的图文件
    sp.save_npz(dst / "A.npz", A_rand)
    sp.save_npz(dst / "L.npz", L_rand)

    print("Saved random graph to:", dst)
    print("A_rand shape:", A_rand.shape, "nnz:", A_rand.nnz)
    print("L_rand shape:", L_rand.shape, "nnz:", L_rand.nnz)
    print("A_rand symmetric?", (A_rand != A_rand.T).nnz == 0)
    print("L_rand symmetric?", (L_rand != L_rand.T).nnz == 0)
    print("diag nnz:", A_rand.diagonal().nonzero()[0].size)
    print("undirected edges:", sp.triu(A_rand, k=1).nnz)


if __name__ == "__main__":
    main()