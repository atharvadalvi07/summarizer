import json
import csv
import os
import warnings
import collections
import math
import random
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

try:
    import community as community_louvain  
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    warnings.warn(
        "python-louvain not found. Falling back to greedy_modularity_communities. "
        "Install with: pip install python-louvain"
    )


INPUT_PATH  = "week8_outputs/enhanced_graph.json"
OUTPUT_DIR  = "week9_outputs"
SEED        = 42
random.seed(SEED)


def load_graph(path: str) -> nx.MultiDiGraph:
    """Load Week 8 node-link JSON"""
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — generating mock graph for testing.")
        return _mock_graph()

    with open(path) as f:
        data = json.load(f)

    G = nx.node_link_graph(data, multigraph=True, directed=True)
    print(f"[OK] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _mock_graph() -> nx.MultiDiGraph:
    """Synthetic MultiDiGraph for offline testing."""
    G = nx.MultiDiGraph()
    clusters = [
        ["neural_network", "deep_learning", "backpropagation", "gradient_descent", "optimizer"],
        ["transformer", "attention_mechanism", "bert", "gpt", "tokenization"],
        ["knowledge_graph", "entity", "relation", "triple", "ontology"],
        ["summarization", "rouge", "bleu", "abstractive", "extractive"],
        ["sciSpacy", "ner", "entity_recognition", "biomedical", "pubmed"],
    ]
    for group in clusters:
        for u in group:
            for v in group:
                if u != v:
                    G.add_edge(u, v, weight=random.uniform(0.5, 2.0), edge_type="co-occurrence")

    cross = [
        ("transformer", "attention_mechanism"),
        ("bert", "summarization"),
        ("entity", "knowledge_graph"),
        ("ner", "entity"),
        ("deep_learning", "transformer"),
    ]
    for u, v in cross:
        G.add_edge(u, v, weight=0.3, edge_type="dependency")
    print(f"[MOCK] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def to_undirected_simple(G: nx.MultiDiGraph) -> nx.Graph:
    """
    Convert to a weighted simple undirected graph for community detection.
    """
    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)
    for n, attrs in G.nodes(data=True):
        if n in UG:
            UG.nodes[n].update(attrs)
    return UG


# COMMUNITY DETECTION

def run_louvain(UG: nx.Graph) -> tuple[dict, float]:
    """Louvain community detection (python-louvain)."""
    partition = community_louvain.best_partition(UG, weight="weight", random_state=SEED)
    modularity = community_louvain.modularity(partition, UG, weight="weight")
    return partition, modularity


def run_greedy_modularity(UG: nx.Graph) -> tuple[dict, float]:
    """NetworkX greedy modularity communities (fallback)."""
    communities = list(nx.community.greedy_modularity_communities(UG, weight="weight"))
    partition = {}
    for cid, members in enumerate(communities):
        for node in members:
            partition[node] = cid
    modularity = nx.community.modularity(UG, communities, weight="weight")
    return partition, modularity


def run_girvan_newman(UG: nx.Graph, max_communities: int = 6) -> tuple[dict, float]:
    """
    Girvan-Newman edge-betweenness community detection.
    """
    comp_gen = nx.community.girvan_newman(UG)
    best_partition = None
    best_mod = -1.0

    for communities in comp_gen:
        if len(communities) > max_communities:
            break
        partition = {}
        for cid, members in enumerate(communities):
            for node in members:
                partition[node] = cid
        mod = nx.community.modularity(UG, communities, weight="weight")
        if mod > best_mod:
            best_mod = mod
            best_partition = partition

    return best_partition, best_mod


def community_summary(partition: dict) -> dict:
    """Return {community_id: [nodes]} mapping."""
    groups = collections.defaultdict(list)
    for node, cid in partition.items():
        groups[cid].append(node)
    return dict(groups)


def inter_community_edges(G: nx.MultiDiGraph, partition: dict) -> list[dict]:
    """List edges that cross community boundaries."""
    cross = []
    for u, v, data in G.edges(data=True):
        if partition.get(u) != partition.get(v):
            cross.append({
                "source": u,
                "target": v,
                "source_community": partition.get(u),
                "target_community": partition.get(v),
                "weight": data.get("weight", 1.0),
                "edge_type": data.get("edge_type", "unknown"),
            })
    return cross


def top_nodes_per_community(G: nx.MultiDiGraph, partition: dict, top_n: int = 5) -> dict:
    """Return top-N nodes by degree centrality within each community."""
    deg_centrality = nx.degree_centrality(G)
    groups = community_summary(partition)
    result = {}
    for cid, nodes in groups.items():
        ranked = sorted(nodes, key=lambda n: deg_centrality.get(n, 0), reverse=True)
        result[cid] = ranked[:top_n]
    return result


def compute_graph_metrics(G: nx.MultiDiGraph, UG: nx.Graph, partition: dict) -> dict:
    """Compile structural + community metrics into a single dict."""
    deg_centrality = nx.degree_centrality(G)
    between_centrality = nx.betweenness_centrality(UG, weight="weight", normalized=True)

    n_communities = len(set(partition.values()))
    communities_as_sets = [
        {n for n, c in partition.items() if c == cid}
        for cid in set(partition.values())
    ]

    intra = sum(
        1 for u, v in G.edges()
        if partition.get(u) == partition.get(v)
    )
    inter = G.number_of_edges() - intra

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "num_communities": n_communities,
        "avg_community_size": round(G.number_of_nodes() / max(n_communities, 1), 2),
        "intra_community_edges": intra,
        "inter_community_edges": inter,
        "avg_degree_centrality": round(sum(deg_centrality.values()) / max(len(deg_centrality), 1), 6),
        "top_degree_central_nodes": sorted(deg_centrality, key=deg_centrality.get, reverse=True)[:10],
        "avg_betweenness_centrality": round(sum(between_centrality.values()) / max(len(between_centrality), 1), 6),
        "top_betweenness_nodes": sorted(between_centrality, key=between_centrality.get, reverse=True)[:10],
    }


# visualization SIMPLIFY


def visualize_communities(G: nx.MultiDiGraph, partition: dict, output_path: str):
    UG_simple = to_undirected_simple(G)
    n_communities = len(set(partition.values()))

    cmap = cm.get_cmap("tab10", n_communities)
    node_colors = [cmap(partition.get(n, 0)) for n in UG_simple.nodes()]

    degrees = dict(UG_simple.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [200 + 800 * (degrees.get(n, 1) / max_deg) for n in UG_simple.nodes()]

    weights = [UG_simple[u][v].get("weight", 1.0) for u, v in UG_simple.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 2.5 * (w / max_w) for w in weights]

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    pos = nx.spring_layout(UG_simple, seed=SEED, k=1.5 / math.sqrt(max(UG_simple.number_of_nodes(), 1)))

    nx.draw_networkx_edges(
        UG_simple, pos,
        width=edge_widths,
        edge_color="#aaaaaa",
        alpha=0.35,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        UG_simple, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )

    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:30]
    labels = {n: n for n in top_nodes if n in UG_simple.nodes()}
    nx.draw_networkx_labels(
        UG_simple, pos,
        labels=labels,
        font_size=7,
        font_color="white",
        ax=ax,
    )

    patches = [
        mpatches.Patch(color=cmap(i), label=f"Community {i}")
        for i in range(n_communities)
    ]
    ax.legend(handles=patches, loc="upper left", framealpha=0.4,
              labelcolor="white", facecolor="#222222", fontsize=8)

    ax.set_title(
        f"Week 9 — Knowledge Graph Communities\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · {n_communities} communities",
        color="white", fontsize=13, pad=12,
    )
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Visualization saved → {output_path}")


def save_outputs(
    partition: dict,
    community_groups: dict,
    metrics: dict,
    top_nodes: dict,
    cross_edges: list,
    louvain_mod: float,
    gn_mod: float,
    greedy_mod: float,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    assignments_path = os.path.join(output_dir, "community_assignments.json")
    payload = {
        "partition": partition,
        "communities": {str(k): v for k, v in community_groups.items()},
        "top_nodes_per_community": {str(k): v for k, v in top_nodes.items()},
        "inter_community_edges": cross_edges[:50],  # cap for readability
    }
    with open(assignments_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Community assignments → {assignments_path}")

    report = {
        **metrics,
        "modularity_scores": {
            "louvain": round(louvain_mod, 4) if louvain_mod is not None else None,
            "greedy_modularity": round(greedy_mod, 4) if greedy_mod is not None else None,
            "girvan_newman": round(gn_mod, 4) if gn_mod is not None else None,
        },
        "community_sizes": {
            str(cid): len(nodes)
            for cid, nodes in community_groups.items()
        },
    }
    report_json_path = os.path.join(output_dir, "analytics_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Analytics report (JSON) → {report_json_path}")

    report_csv_path = os.path.join(output_dir, "analytics_report.csv")
    flat_rows = [
        {"metric": k, "value": (", ".join(v) if isinstance(v, list) else v)}
        for k, v in report.items()
        if not isinstance(v, dict)
    ]
    for nested_key in ("modularity_scores", "community_sizes"):
        if nested_key in report:
            for k, v in report[nested_key].items():
                flat_rows.append({"metric": f"{nested_key}.{k}", "value": v})

    with open(report_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[OK] Analytics report (CSV) → {report_csv_path}")



def main():
    print("=" * 60)
    print("Week 9 — Community Detection & Graph Analytics")
    print("=" * 60)

    G  = load_graph(INPUT_PATH)
    UG = to_undirected_simple(G)

    louvain_partition = greedy_partition = gn_partition = None
    louvain_mod = greedy_mod = gn_mod = None

    if LOUVAIN_AVAILABLE:
        print("\n[1/3] Running Louvain community detection …")
        louvain_partition, louvain_mod = run_louvain(UG)
        print(f"      Communities: {len(set(louvain_partition.values()))}  |  Modularity: {louvain_mod:.4f}")
    else:
        print("\n[1/3] Louvain skipped (library not installed).")

    print("\n[2/3] Running Greedy Modularity communities …")
    greedy_partition, greedy_mod = run_greedy_modularity(UG)
    print(f"      Communities: {len(set(greedy_partition.values()))}  |  Modularity: {greedy_mod:.4f}")

    print("\n[3/3] Running Girvan-Newman (max 6 communities) …")
    try:
        gn_partition, gn_mod = run_girvan_newman(UG, max_communities=6)
        print(f"      Communities: {len(set(gn_partition.values()))}  |  Modularity: {gn_mod:.4f}")
    except Exception as e:
        print(f"      [WARN] Girvan-Newman failed: {e}")
        gn_partition, gn_mod = greedy_partition, greedy_mod

    # Select best partition
    candidates = [
        (louvain_mod, louvain_partition, "Louvain"),
        (greedy_mod,  greedy_partition,  "Greedy Modularity"),
        (gn_mod,      gn_partition,      "Girvan-Newman"),
    ]
    best_mod, best_partition, best_method = max(
        (c for c in candidates if c[0] is not None),
        key=lambda x: x[0],
    )
    print(f"\n[BEST] {best_method} selected (modularity = {best_mod:.4f})")

    community_groups = community_summary(best_partition)
    top_nodes        = top_nodes_per_community(G, best_partition)
    cross_edges      = inter_community_edges(G, best_partition)
    metrics          = compute_graph_metrics(G, UG, best_partition)

    print("\n── Graph Metrics ────────────────────────────────────────")
    for k, v in metrics.items():
        if not isinstance(v, list):
            print(f"   {k:40s}: {v}")

    print("\n── Saving outputs ───────────────────────────────────────")
    save_outputs(
        partition=best_partition,
        community_groups=community_groups,
        metrics=metrics,
        top_nodes=top_nodes,
        cross_edges=cross_edges,
        louvain_mod=louvain_mod,
        gn_mod=gn_mod,
        greedy_mod=greedy_mod,
        output_dir=OUTPUT_DIR,
    )

    print("\n── Generating visualization ─────────────────────────────")
    visualize_communities(G, best_partition, os.path.join(OUTPUT_DIR, "week9_graph.png"))

    print("\n✓ Week 9 complete. Outputs in:", OUTPUT_DIR)
    print("  community_assignments.json")
    print("  analytics_report.json")
    print("  analytics_report.csv")
    print("  week9_graph.png")


if __name__ == "__main__":
    main()