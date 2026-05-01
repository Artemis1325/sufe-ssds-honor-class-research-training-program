import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True, help="Dir with X.npy/y.npy/meta.json")
    p.add_argument("--string_links", type=str, required=True, help="10090.protein.links...txt(.gz)")
    p.add_argument("--string_aliases", type=str, required=True, help="10090.protein.aliases...txt(.gz)")
    p.add_argument("--score_min", type=float, default=400, help="min combined_score in [0,1000]")
    p.add_argument("--undirected", action="store_true", help="make graph symmetric")
    p.add_argument("--chunksize", type=int, default=2_000_000, help="links file chunk size")
    return p.parse_args()


def normalize_feat(name: str) -> str:
    # remove suffix
    if name.endswith("_N"):
        name = name[:-2]

    # phospho features: pAKT -> AKT
    if name.startswith("p") and len(name) > 1 and name[1:].isalnum():
        # only strip if p+letters/digits; safe for your feature names
        name = name[1:]

    # Mouse-style symbol: DYRK1A -> Dyrk1a
    if name.isupper():
        name = name.capitalize()

    return name



def build_alias_to_pid_map(aliases_path: Path):
    """
    Build alias -> string_protein_id mapping.
    Handle collisions by keeping the first occurrence (good enough for 77 features),
    and also report collisions for debugging.
    """
    df = pd.read_csv(
        aliases_path,
        sep="\t",
        comment="#",
        header=None,
        names=["pid", "alias", "source"],
        engine="python",
        compression="infer",
        on_bad_lines="skip",
    )

    alias_to_pid = {}
    collisions = 0
    for pid, alias in zip(df["pid"].astype(str).values, df["alias"].astype(str).values):
        if alias not in alias_to_pid:
            alias_to_pid[alias] = pid
        else:
            if alias_to_pid[alias] != pid:
                collisions += 1
    print(f"[alias] map size={len(alias_to_pid)} collisions_seen={collisions}")
    return alias_to_pid


def main():
    args = parse_args()
    ddir = Path(args.dataset_dir).resolve()
    meta_path = ddir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_names = meta["feature_names"]
    feat_norm = [normalize_feat(x) for x in feat_names]

    # Build alias map
    alias_to_pid = build_alias_to_pid_map(Path(args.string_aliases).resolve())

    # Try direct alias match for features (gene symbols)
    feat_to_pid = {}
    # optional: family/canonical candidates (small set, expandable)
    FAMILY_CANDIDATES = {
        "Akt": ["Akt", "Akt1", "Akt2", "Akt3"],
        "Creb": ["Creb", "Creb1", "Creb2"],
        "Camkii": ["Camkii", "Camk2a", "Camk2b", "Camk2d", "Camk2g"],
        "Jnk": ["Jnk", "Mapk8", "Mapk9", "Mapk10"],
        "Mek": ["Mek", "Map2k1", "Map2k2"],
        "Rsk": ["Rsk", "Rps6ka1", "Rps6ka2", "Rps6ka3", "Rps6ka4", "Rps6ka5", "Rps6ka6"],
    }

    for f in feat_norm:
        # try direct and common variants
        candidates = [f, f.capitalize(), f.upper(), f.lower()]

        # add family candidates if applicable
        if f in FAMILY_CANDIDATES:
            candidates = FAMILY_CANDIDATES[f] + candidates

        pid = None
        for c in candidates:
            if c in alias_to_pid:
                pid = alias_to_pid[c]
                break

        if pid is not None:
            feat_to_pid[f] = pid

    keep_idx = [i for i, f in enumerate(feat_norm) if f in feat_to_pid]
    dropped = [f for f in feat_norm if f not in feat_to_pid]

    print(f"[align] features before={len(feat_norm)} mapped={len(keep_idx)} dropped={len(dropped)}")
    if len(keep_idx) == 0:
        # Provide actionable hint
        sample_alias = list(alias_to_pid.keys())[:20]
        raise RuntimeError(
            "No features mapped via alias file. "
            "This likely means your feature names are not gene symbols. "
            f"Example aliases: {sample_alias}"
        )

    # Slice X to aligned features
    X = np.load(ddir / "X.npy")
    y = np.load(ddir / "y.npy")
    X2 = X[:, keep_idx]
    np.save(ddir / "X.npy", X2)  # overwrite X to aligned X
    # update feature list in meta to aligned version (keep normalized names)
    kept_feat = [feat_norm[i] for i in keep_idx]
    meta["feature_names_raw"] = feat_names
    meta["feature_names"] = kept_feat
    meta["aligned_drop_features"] = dropped

    # Create pid->new index
    kept_pid = [feat_to_pid[f] for f in kept_feat]
    pid_to_new = {pid: j for j, pid in enumerate(kept_pid)}
    n = len(kept_pid)

    # Read links file in chunks (space-separated)
    links_path = Path(args.string_links).resolve()
    rows, cols, data = [], [], []
    hit = 0

    it = pd.read_csv(
        links_path,
        sep=" ",
        header=0,
        engine="python",
        chunksize=args.chunksize,
        compression="infer",
    )

    for chunk in it:
        p1 = chunk.iloc[:, 0].astype(str).values
        p2 = chunk.iloc[:, 1].astype(str).values
        sc = chunk.iloc[:, 2].astype(float).values

        mask = sc >= args.score_min
        p1 = p1[mask]; p2 = p2[mask]; sc = sc[mask] / 1000.0

        for a, b, w in zip(p1, p2, sc):
            ia = pid_to_new.get(a, None)
            ib = pid_to_new.get(b, None)
            if ia is None or ib is None or ia == ib:
                continue
            rows.append(ia); cols.append(ib); data.append(w)
            if args.undirected:
                rows.append(ib); cols.append(ia); data.append(w)
            hit += 1

    if hit == 0:
        raise RuntimeError(
            "No edges matched mapped proteins. "
            "Try lowering --score_min (e.g., 200/150), or check mapping quality."
        )

    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()

    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    L = sp.eye(n, format="csr") - (D_inv_sqrt @ W @ D_inv_sqrt)
    sp.save_npz(ddir / "L.npz", L)

    meta["string"] = {
        "links": str(links_path),
        "aliases": str(Path(args.string_aliases).resolve()),
        "score_min": float(args.score_min),
        "nodes": int(n),
        "edges": int(W.nnz // 2 if args.undirected else W.nnz),
        "density": float(W.nnz) / float(n * n),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[graph] X aligned:", X.shape, "->", X2.shape)
    print("[graph] Saved L.npz:", ddir / "L.npz")
    print("[graph] nodes:", n, "edges:", meta["string"]["edges"], "density:", meta["string"]["density"])
    if len(dropped) > 0:
        print("[align] dropped features (first 20):", dropped[:20])


if __name__ == "__main__":
    main()

