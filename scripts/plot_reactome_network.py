from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REACTOME_EDGE_FILE = ROOT / "datasets" / "tcga_brca" / "reactome_edges_symbol.tsv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Prepared dataset directory with feature_graph_symbols.txt")
    p.add_argument("--out-dir", required=True, help="Directory to save the figure and stats")
    p.add_argument("--title", required=True, help="Figure title")
    p.add_argument("--prefix", default="reactome_structure", help="Output filename prefix")
    p.add_argument("--focus-n", type=int, default=20, help="Number of hub genes to keep in the focus subgraph")
    p.add_argument("--label-top-n", type=int, default=20, help="Number of hub labels to show in the focus panel")
    return p.parse_args()


def load_graph(data_dir: Path) -> nx.Graph:
    genes = [
        line.strip()
        for line in (data_dir / "feature_graph_symbols.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gene_set = set(genes)
    edges = pd.read_csv(REACTOME_EDGE_FILE, sep="\t")
    edges = edges[edges["gene_a"].isin(gene_set) & edges["gene_b"].isin(gene_set)].copy()
    return nx.from_pandas_edgelist(edges, "gene_a", "gene_b")


def make_overview_subgraph(graph: nx.Graph) -> nx.Graph:
    largest_cc_nodes = max(nx.connected_components(graph), key=len)
    largest_cc = graph.subgraph(largest_cc_nodes).copy()
    return nx.k_core(largest_cc, k=3)


def make_focus_subgraph(graph: nx.Graph, focus_n: int) -> nx.Graph:
    focus_nodes = []
    for node, _ in sorted(graph.degree, key=lambda x: x[1], reverse=True):
        # Histone families often dominate degree rankings while being less readable in figure labels.
        if node.startswith("H3") or node.startswith("H4"):
            continue
        focus_nodes.append(node)
        if len(focus_nodes) >= focus_n:
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
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(data_dir)
    overview = make_overview_subgraph(graph)
    focus = make_focus_subgraph(graph, args.focus_n)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    plt.subplots_adjust(wspace=0.08)

    draw_graph(
        axes[0],
        overview,
        title=f"Reactome overview (3-core, {overview.number_of_nodes()} nodes)",
        label_top_n=0,
    )
    nodes1 = draw_graph(
        axes[1],
        focus,
        title=f"Representative hub subnetwork ({focus.number_of_nodes()} nodes)",
        label_top_n=args.label_top_n,
    )

    cbar = fig.colorbar(nodes1, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    cbar.set_label("Node degree", rotation=90)
    fig.suptitle(args.title, fontsize=14, y=0.98)

    fig_path = out_dir / f"{args.prefix}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
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
    stats_path = out_dir / f"{args.prefix}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved figure to: {fig_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
