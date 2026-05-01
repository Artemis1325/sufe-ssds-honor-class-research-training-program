from pathlib import Path
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "colonomics_crc_allnormal_vs_tumor_reactome"
REACTOME_EDGE_FILE = ROOT / "datasets" / "tcga_brca" / "reactome_edges_symbol.tsv"
OUT_DIR = ROOT / "results" / "colonomics_reactome_network"


def load_graph() -> nx.Graph:
    genes = [line.strip() for line in (DATA_DIR / "feature_graph_symbols.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    gene_set = set(genes)
    edges = pd.read_csv(REACTOME_EDGE_FILE, sep="\t")
    edges = edges[edges["gene_a"].isin(gene_set) & edges["gene_b"].isin(gene_set)].copy()
    return nx.from_pandas_edgelist(edges, "gene_a", "gene_b")


def make_overview_subgraph(graph: nx.Graph) -> nx.Graph:
    largest_cc_nodes = max(nx.connected_components(graph), key=len)
    largest_cc = graph.subgraph(largest_cc_nodes).copy()
    return nx.k_core(largest_cc, k=3)


def make_focus_subgraph(graph: nx.Graph) -> nx.Graph:
    focus_nodes = []
    for node, _ in sorted(graph.degree, key=lambda x: x[1], reverse=True):
        if node.startswith("H3") or node.startswith("H4"):
            continue
        focus_nodes.append(node)
        if len(focus_nodes) >= 20:
            break
    return graph.subgraph(focus_nodes).copy()


def draw_graph(ax, graph: nx.Graph, title: str, label_top_n: int = 0):
    degrees = dict(graph.degree())
    degree_values = np.array([degrees[n] for n in graph.nodes()], dtype=float)
    node_sizes = 10 + 2.0 * degree_values
    pos = nx.spring_layout(graph, seed=42, k=0.24 / np.sqrt(max(graph.number_of_nodes(), 2)), iterations=250)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#9AA5B1", width=0.4, alpha=0.18)
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=degree_values,
        cmap=plt.cm.YlGnBu,
        linewidths=0.2,
        edgecolors="#6B7280",
    )

    if label_top_n > 0:
        top_nodes = [node for node, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:label_top_n]]
        nx.draw_networkx_labels(
            graph,
            {node: pos[node] for node in top_nodes},
            labels={node: node for node in top_nodes},
            font_size=8,
            font_weight="bold",
            font_color="#111827",
            ax=ax,
        )

    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    return nodes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    overview = make_overview_subgraph(graph)
    focus = make_focus_subgraph(graph)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    plt.subplots_adjust(wspace=0.08)

    draw_graph(
        axes[0],
        overview,
        title=f"CRC-Reactome overview (3-core, {overview.number_of_nodes()} nodes)",
        label_top_n=0,
    )
    nodes1 = draw_graph(
        axes[1],
        focus,
        title=f"Representative hub subnetwork ({focus.number_of_nodes()} nodes)",
        label_top_n=20,
    )

    cbar = fig.colorbar(nodes1, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    cbar.set_label("Node degree", rotation=90)
    fig.suptitle("Colonomics CRC Reactome prior graph", fontsize=14, y=0.98)
    fig.savefig(OUT_DIR / "colonomics_reactome_structure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_components": nx.number_connected_components(graph),
        "largest_component_nodes": len(max(nx.connected_components(graph), key=len)),
        "overview_nodes": overview.number_of_nodes(),
        "overview_edges": overview.number_of_edges(),
        "focus_nodes": focus.number_of_nodes(),
        "focus_edges": focus.number_of_edges(),
        "top_degree_nodes": sorted(graph.degree, key=lambda x: x[1], reverse=True)[:20],
    }
    with open(OUT_DIR / "colonomics_reactome_structure_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved figure to: {OUT_DIR / 'colonomics_reactome_structure.png'}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
