import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_edge_list_from_adj(A: sp.spmatrix):
    A = A.tocsr()
    A = sp.triu(A, k=1).tocsr()
    rows, cols = A.nonzero()
    edges = [(int(u), int(v)) for u, v in zip(rows, cols)]
    return edges


def edge_set_from_list(edges):
    return set(edges)


def try_swap(e1, e2, edge_set):
    a, b = e1
    c, d = e2

    # 四个端点必须互不相同，否则容易产生自环/退化交换
    if len({a, b, c, d}) < 4:
        return None

    # 两种交换方式里随机试一种
    if np.random.rand() < 0.5:
        ne1 = (min(a, d), max(a, d))
        ne2 = (min(c, b), max(c, b))
    else:
        ne1 = (min(a, c), max(a, c))
        ne2 = (min(d, b), max(d, b))

    # 禁止自环
    if ne1[0] == ne1[1] or ne2[0] == ne2[1]:
        return None

    # 新边不能和旧边重复，也不能彼此重复
    if ne1 == ne2:
        return None

    old1 = (min(a, b), max(a, b))
    old2 = (min(c, d), max(c, d))

    # 先临时移除旧边，再检查新边是否已存在
    edge_set.remove(old1)
    edge_set.remove(old2)

    valid = (ne1 not in edge_set) and (ne2 not in edge_set)

    if valid:
        edge_set.add(ne1)
        edge_set.add(ne2)
        return ne1, ne2
    else:
        edge_set.add(old1)
        edge_set.add(old2)
        return None


def degree_preserving_rewire(edges, nswap, seed=42, verbose_every=50000):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    edges = [(min(u, v), max(u, v)) for u, v in edges]
    edge_set = edge_set_from_list(edges)
    edges = list(edge_set)

    m = len(edges)
    success = 0
    attempts = 0
    max_attempts = max(10 * nswap, 100000)

    while success < nswap and attempts < max_attempts:
        i, j = rng.integers(0, m, size=2)
        if i == j:
            attempts += 1
            continue

        e1 = edges[i]
        e2 = edges[j]

        swapped = try_swap(e1, e2, edge_set)
        attempts += 1

        if swapped is not None:
            ne1, ne2 = swapped
            edges[i] = ne1
            edges[j] = ne2
            success += 1

            if verbose_every and success % verbose_every == 0:
                print(f"successful swaps: {success}/{nswap}")

    return edges, success, attempts


def edge_list_to_adj(n_nodes, edges):
    rows = []
    cols = []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])

    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    A.sum_duplicates()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def compute_laplacian(A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - A
    return L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets/tcga_brca")
    parser.add_argument("--output_dir", type=str, default="datasets/tcga_brca_degnull")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--swap_factor",
        type=float,
        default=10.0,
        help="number of successful swaps = swap_factor * num_edges",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = sp.load_npz(in_dir / "A.npz").tocsr()
    X = np.load(in_dir / "X_graph.npy")
    y = np.load(in_dir / "y.npy")

    feature_names_path = in_dir / "feature_names.json"
    if feature_names_path.exists():
        feature_names_bytes = feature_names_path.read_bytes()

    n_nodes = A.shape[0]
    edges = load_edge_list_from_adj(A)
    m = len(edges)
    nswap = int(args.swap_factor * m)

    print("input graph:")
    print("  num_nodes =", n_nodes)
    print("  num_edges =", m)
    deg0 = np.asarray(A.sum(axis=1)).ravel()
    print("  min_deg   =", deg0.min())
    print("  max_deg   =", deg0.max())
    print("  mean_deg  =", deg0.mean())

    rewired_edges, success, attempts = degree_preserving_rewire(
        edges=edges,
        nswap=nswap,
        seed=args.seed,
        verbose_every=50000,
    )

    A_new = edge_list_to_adj(n_nodes, rewired_edges)
    L_new = compute_laplacian(A_new)

    deg1 = np.asarray(A_new.sum(axis=1)).ravel()

    print("\nrewired graph:")
    print("  successful_swaps =", success)
    print("  attempts         =", attempts)
    print("  num_edges        =", sp.triu(A_new, k=1).nnz)
    print("  min_deg          =", deg1.min())
    print("  max_deg          =", deg1.max())
    print("  mean_deg         =", deg1.mean())
    print("  degree_preserved =", np.array_equal(np.sort(deg0), np.sort(deg1)))

    sp.save_npz(out_dir / "A.npz", A_new)
    sp.save_npz(out_dir / "L.npz", L_new)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    if feature_names_path.exists():
        (out_dir / "feature_names.json").write_bytes(feature_names_bytes)

    print(f"\nsaved to: {out_dir}")


if __name__ == "__main__":
    main()