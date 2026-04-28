import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import spacy


def load_nlp_model(model_name: str = "en_core_sci_sm"):
    """Load SciSpacy model; fallback to en_core_web_sm for quick testing."""
    try:
        nlp = spacy.load(model_name)
        print(f"[INFO] Loaded model: {model_name}")
    except OSError:
        print(f"[WARN] {model_name} not found. Falling back to en_core_web_sm.")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def build_cooccurrence_edges(
    entity_sentences: list[tuple[str, list[str]]],
    window_size: int = 2,
    min_weight: int = 2,
) -> dict[tuple[str, str], int]:
    """
    Parameters
    ----------
    entity_sentences : list of (sentence_text, [entity1, entity2, ...])
    window_size      : how many adjacent sentences to consider together
   
    """
    edge_weights: dict[tuple[str, str], int] = defaultdict(int)

    for i in range(len(entity_sentences)):
        window_entities: list[str] = []
        for j in range(i, min(i + window_size, len(entity_sentences))):
            _, ents = entity_sentences[j]
            window_entities.extend(ents)

       
        unique_in_window = list(set(window_entities))

       
        for a, b in combinations(sorted(unique_in_window), 2):
            if a != b:
                edge_weights[(a, b)] += 1

    pruned = {k: v for k, v in edge_weights.items() if v >= min_weight}
    if not pruned:
        print(f"[WARN] All edges pruned — returning raw edges. Consider lowering min_weight.")
        return dict(edge_weights)
    print(f"[INFO] Co-occurrence edges: {len(edge_weights)} raw → {len(pruned)} after pruning (min_weight={min_weight})")
    return pruned


def extract_svo_triples(
    texts: list[str],
    nlp,
    entity_set: set[str],
) -> list[dict]:
   
    triples = []

    for text in texts:
        doc = nlp(text)
        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            subj = obj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child.text.strip()
                if child.dep_ in ("dobj", "pobj", "attr", "ccomp"):
                    obj = child.text.strip()

            if subj and obj:
                subj_canon = _match_entity(subj, entity_set)
                obj_canon = _match_entity(obj, entity_set)

                if subj_canon and obj_canon and subj_canon != obj_canon:
                    triples.append({
                        "subj": subj_canon,
                        "verb": token.lemma_,
                        "obj": obj_canon,
                        "sentence": text[:120],
                    })

    print(f"[INFO] SVO triples extracted: {len(triples)}")
    return triples


def _match_entity(token_text: str, entity_set: set[str]) -> str | None:
    token_lower = token_text.lower()
    for ent in entity_set:
        if token_lower in ent.lower() or ent.lower() in token_lower:
            return ent
    return None



def build_enhanced_graph(
    entities: list[str],
    cooccurrence_weights: dict[tuple[str, str], int],
    svo_triples: list[dict],
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for ent in entities:
        G.add_node(ent)

    
    for (a, b), weight in cooccurrence_weights.items():
        G.add_edge(a, b, edge_type="cooccurrence", weight=weight)
        G.add_edge(b, a, edge_type="cooccurrence", weight=weight)

    
    for triple in svo_triples:
        G.add_edge(
            triple["subj"],
            triple["obj"],
            edge_type="dependency",
            relation=triple["verb"],
            weight=1,
            sentence=triple["sentence"],
        )

    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def normalize_cooccurrence_weights(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    node_freq: dict[str, float] = defaultdict(float)
    total = 0.0

    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "cooccurrence":
            w = data.get("weight", 1)
            node_freq[u] += w
            node_freq[v] += w
            total += w

    if total == 0:
        return G

    import math
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("edge_type") == "cooccurrence":
            p_ab = data["weight"] / total
            p_a = node_freq[u] / total
            p_b = node_freq[v] / total
            pmi = math.log(p_ab / (p_a * p_b + 1e-9) + 1e-9)
            G[u][v][key]["pmi"] = round(pmi, 4)

    print("[INFO] PMI normalization applied to co-occurrence edges.")
    return G



def compute_metrics_report(G: nx.MultiDiGraph) -> pd.DataFrame:
    cooc_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "cooccurrence"]
    dep_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "dependency"]

    G_undirected = nx.Graph(G.to_undirected())

    metrics = {
        "Total Nodes": G.number_of_nodes(),
        "Total Edges": G.number_of_edges(),
        "Co-occurrence Edges": len(cooc_edges),
        "Dependency Edges": len(dep_edges),
        "Graph Density": round(nx.density(G), 6),
        "Avg In-Degree": round(sum(d for _, d in G.in_degree()) / max(G.number_of_nodes(), 1), 3),
        "Avg Out-Degree": round(sum(d for _, d in G.out_degree()) / max(G.number_of_nodes(), 1), 3),
        "Weakly Connected Components": nx.number_weakly_connected_components(G),
        "Largest WCC Size": len(max(nx.weakly_connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0,
        "Avg Clustering Coeff (undirected)": round(nx.average_clustering(G_undirected), 4),
    }

    deg_centrality = nx.degree_centrality(G_undirected)
    top5 = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    metrics["Top-5 Central Nodes"] = ", ".join(f"{n} ({v:.3f})" for n, v in top5)

    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    return df


def visualize_graph(G: nx.MultiDiGraph, output_path: str = "week8_graph.png"):
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, k=1.5)

    cooc_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "cooccurrence"]
    dep_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "dependency"]

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="#AED6F1", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")

    nx.draw_networkx_edges(G, pos, edgelist=cooc_edges,
                           edge_color="#2196F3", alpha=0.5, width=1.0,
                           arrows=False, label="Co-occurrence")
    nx.draw_networkx_edges(G, pos, edgelist=dep_edges,
                           edge_color="#E53935", alpha=0.7, width=1.5,
                           arrows=True, arrowsize=12, label="Dependency")


    dep_labels = {
        (u, v): d.get("relation", "")
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "dependency"
    }
    if len(dep_labels) <= 15:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=dep_labels, font_size=6, label_pos=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", lw=2, label="Co-occurrence"),
        Line2D([0], [0], color="#E53935", lw=2, label="Dependency (SVO)"),
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    plt.title("Week 8 — Enhanced Knowledge Graph\n(Co-occurrence + Dependency Edges)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Graph visualization saved → {output_path}")




def save_outputs(G: nx.MultiDiGraph, metrics_df: pd.DataFrame, out_dir: str = "week8_outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save graph as JSON 
    graph_data = nx.node_link_data(G)
    graph_path = Path(out_dir) / "enhanced_graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"[INFO] Graph saved → {graph_path}")

    # Metrics
    metrics_path = Path(out_dir) / "metrics_report.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Metrics report saved → {metrics_path}")

    
    G_simple = nx.DiGraph()
    for node in G.nodes():
        G_simple.add_node(node)
    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "unknown")
        weight = data.get("weight", 1)
        relation = data.get("relation", "")
        G_simple.add_edge(u, v, edge_type=edge_type, weight=weight, relation=relation)

    graphml_path = Path(out_dir) / "enhanced_graph.graphml"
    nx.write_graphml(G_simple, str(graphml_path))
    print(f"[INFO] GraphML saved → {graphml_path}")




def run_week8_pipeline(
    entities_path:  str,          # canonical_entities.json  (Week 6 output)
    cleaned_path:   str,          # cleaned_dataset.json     (raw text source)
    graph_json:     str,          # output: enhanced_graph.json
    graph_graphml:  str,          # output: enhanced_graph.graphml
    metrics_csv:    str,          # output: metrics_report.csv
    viz_path:       str,          # output: week8_graph.png
    cooc_window:    int = 2,
    cooc_min_weight: int = 2,
):
    print(f"\n{'='*55}")
    print("  Week 8 — Enhanced Edge Construction")
    print(f"{'='*55}\n")

    out_dir = str(Path(graph_json).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    entity_sentences: list[tuple[str, list[str]]] = []
    all_entity_strings: list[str] = []

    if Path(entities_path).exists():
        with open(entities_path, encoding="utf-8") as f:
            canonical_data = json.load(f)
        if isinstance(canonical_data, dict):
            entries = canonical_data.get("canonical_entities", [])
            for entry in entries:
                if isinstance(entry, dict):
                    text = (entry.get("normalized_text")
                            or entry.get("text")
                            or entry.get("canonical")
                            or "")
                    all_entity_strings.append(text)
                elif isinstance(entry, str):
                    all_entity_strings.append(entry)
        elif isinstance(canonical_data, list):
            for entry in canonical_data:
                if isinstance(entry, dict):
                    text = (entry.get("normalized_text")
                            or entry.get("text")
                            or entry.get("canonical")
                            or "")
                    all_entity_strings.append(text)
                elif isinstance(entry, str):
                    all_entity_strings.append(entry)

        all_entity_strings = [e for e in all_entity_strings if e]
        print(f"[INFO] Loaded {len(all_entity_strings)} canonical entities from {entities_path}")
    else:
        print(f"[WARN] entities_path not found: {entities_path} — using demo entities.")

    

    if Path(cleaned_path).exists():
        with open(cleaned_path, encoding="utf-8") as f:
            cleaned_data = json.load(f)
        for paper in (cleaned_data if isinstance(cleaned_data, list) else []):
            text = (paper.get("abstract_clean")
                    or paper.get("abstract_raw")
                    or paper.get("abstract")
                    or paper.get("text")
                    or "")
            if text:
                raw_texts.append(text)
        print(f"[INFO] Loaded {len(raw_texts)} paper texts from {cleaned_path}")
    else:
        print(f"[WARN] cleaned_path not found: {cleaned_path} — SVO triples will be empty.")

    paper_sentences: list[str] = []
    if Path(cleaned_path).exists():
        with open(cleaned_path, encoding="utf-8") as _f:
            _cleaned2 = json.load(_f)
        for _paper in (_cleaned2 if isinstance(_cleaned2, list) else []):
            _sents = _paper.get("sentences", [])
            if isinstance(_sents, list):
                paper_sentences.extend([s for s in _sents if isinstance(s, str) and s.strip()])

    if all_entity_strings and raw_texts:
        entity_set_lower = {e.lower(): e for e in all_entity_strings}

        all_sentences = paper_sentences if paper_sentences else [
            s.strip()
            for text in raw_texts
            for s in text.replace("\n", " ").split(".")
            if s.strip()
        ]
        print(f"[INFO] Matching entities across {len(all_sentences)} sentences")

        for sent in all_sentences:
            sent_lower = sent.lower()
            found = [canon for lower, canon in entity_set_lower.items() if lower in sent_lower]
            if found:
                entity_sentences.append((sent, found))
    elif not all_entity_strings:
        # Demo fallback if code keeps crashing
        print("[INFO] Using demo entity_sentences (no canonical entities loaded).")
        entity_sentences = [
            ("BERT is used for NER tasks.", ["BERT", "NER"]),
            ("SciSpacy applies NER to biomedical text.", ["SciSpacy", "NER", "biomedical text"]),
            ("BioBERT improves BERT for biomedical NLP.", ["BioBERT", "BERT", "biomedical NLP"]),
            ("NER extracts entities from PubMed abstracts.", ["NER", "PubMed"]),
            ("Knowledge graphs represent entity relationships.", ["knowledge graphs", "entities"]),
            ("NetworkX builds knowledge graphs in Python.", ["NetworkX", "knowledge graphs", "Python"]),
            ("Entity normalization reduces duplicates.", ["entity normalization", "entities"]),
            ("RapidFuzz performs fuzzy entity matching.", ["RapidFuzz", "entity normalization"]),
        ]
        all_entity_strings = list({e for _, ents in entity_sentences for e in ents})
        raw_texts = [s for s, _ in entity_sentences]

    if not entity_sentences:
        print("[WARN] No entity_sentences built — pipeline will produce an empty graph.")

    entity_set = set(all_entity_strings)

    nlp = load_nlp_model()

    cooc_weights = build_cooccurrence_edges(
        entity_sentences,
        window_size=int(cooc_window),
        min_weight=int(cooc_min_weight),
    )
    svo_triples = extract_svo_triples(raw_texts, nlp, entity_set)
    G = build_enhanced_graph(list(entity_set), cooc_weights, svo_triples)
    G = normalize_cooccurrence_weights(G)
    metrics_df = compute_metrics_report(G)

    print("\n--- Graph Metrics Report ---")
    print(metrics_df.to_string(index=False))

    visualize_graph(G, output_path=viz_path)

    Path(graph_json).parent.mkdir(parents=True, exist_ok=True)
    with open(graph_json, "w") as f:
        json.dump(nx.node_link_data(G), f, indent=2)
    print(f"[INFO] Graph JSON saved → {graph_json}")

    Path(metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[INFO] Metrics CSV saved → {metrics_csv}")

    G_simple = nx.DiGraph()
    for node in G.nodes():
        G_simple.add_node(node)
    for u, v, data in G.edges(data=True):
        G_simple.add_edge(u, v,
                          edge_type=data.get("edge_type", "unknown"),
                          weight=data.get("weight", 1),
                          relation=data.get("relation", ""))
    Path(graph_graphml).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G_simple, str(graph_graphml))
    print(f"[INFO] GraphML saved → {graph_graphml}")

    print(f"\n[DONE] All Week 8 outputs saved to '{out_dir}/'")
    return G, metrics_df


if __name__ == "__main__":
    G, metrics = run_week8_pipeline(
        entities_path="week6_outputs/canonical_entities.json",
        cleaned_path="data/cleaned_dataset.json",
        graph_json="week8_outputs/enhanced_graph.json",
        graph_graphml="week8_outputs/enhanced_graph.graphml",
        metrics_csv="week8_outputs/metrics_report.csv",
        viz_path="week8_outputs/week8_graph.png",
    )
