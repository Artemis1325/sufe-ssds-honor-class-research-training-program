from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "tcga_brca"
RESULTS_DIR = ROOT / "results"

COMPARE_DIRS = [
    RESULTS_DIR / "gene_compare_brca_real_multiseed20_vs_brca_drop10_s42_multiseed20",
    RESULTS_DIR / "gene_compare_brca_real_multiseed20_vs_brca_drop30_s42_multiseed20",
    RESULTS_DIR / "gene_compare_brca_real_multiseed20_vs_brca_drop50_s42_multiseed20",
]

EDGE_FILE = DATA_DIR / "reactome_edges_symbol_brca.tsv"


def load_real_graph(edge_file: Path):
    df = pd.read_csv(edge_file, sep="\t")
    c1, c2 = df.columns[:2]
    g = nx.Graph()
    for _, row in df.iterrows():
        a = str(row[c1]).strip()
        b = str(row[c2]).strip()
        if a and b and a != "nan" and b != "nan" and a != b:
            g.add_edge(a, b)
    return g


def read_gene_list(tsv_file: Path):
    df = pd.read_csv(tsv_file, sep="\t")
    return df["gene"].tolist()


def summarize_set(g: nx.Graph, genes):
    genes_in_graph = [x for x in genes if x in g]
    sub = g.subgraph(genes_in_graph).copy()
    n = sub.number_of_nodes()
    m = sub.number_of_edges()
    density = 0.0 if n < 2 else nx.density(sub)
    degs = sorted(sub.degree, key=lambda x: (-x[1], x[0]))
    top_hubs = degs[:10]
    return {
        "n_input": len(genes),
        "n_in_graph": n,
        "internal_edges": m,
        "density": density,
        "top_hubs": top_hubs,
    }


def main():
    g = load_real_graph(EDGE_FILE)
    print(f"[real graph] nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")
    print("=" * 100)

    for compare_dir in COMPARE_DIRS:
        real_fp = compare_dir / "real_biased_top200_from_all.tsv"
        drop_fp = compare_dir / "drop_biased_top200_from_all.tsv"

        if not real_fp.exists():
            raise FileNotFoundError(f"Missing file: {real_fp}")
        if not drop_fp.exists():
            raise FileNotFoundError(f"Missing file: {drop_fp}")

        real_genes = read_gene_list(real_fp)
        drop_genes = read_gene_list(drop_fp)

        real_sum = summarize_set(g, real_genes)
        drop_sum = summarize_set(g, drop_genes)

        print(f"[compare_dir] {compare_dir.name}")
        print("-" * 100)

        print("[real-biased set]")
        print(f"input_genes = {real_sum['n_input']}")
        print(f"in_graph_genes = {real_sum['n_in_graph']}")
        print(f"internal_edges = {real_sum['internal_edges']}")
        print(f"density = {real_sum['density']:.6f}")
        print("top_hubs =", ", ".join([f"{gname}({deg})" for gname, deg in real_sum["top_hubs"]]))

        print("-" * 100)

        print("[drop-biased set]")
        print(f"input_genes = {drop_sum['n_input']}")
        print(f"in_graph_genes = {drop_sum['n_in_graph']}")
        print(f"internal_edges = {drop_sum['internal_edges']}")
        print(f"density = {drop_sum['density']:.6f}")
        print("top_hubs =", ", ".join([f"{gname}({deg})" for gname, deg in drop_sum["top_hubs"]]))

        print("=" * 100)


if __name__ == "__main__":
    main()