import argparse
import sys
import os

ROOT = os.path.dirname(__file__)
for week in ["week1", "week2", "week3", "week4", "week5", "week6", "week7", "week8", "week9"]:
    sys.path.insert(0, os.path.join(ROOT, week))

DATA_DIR    = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

# Week-specific output subdirectories
W6_DIR = os.path.join(OUTPUTS_DIR, "week6_outputs")
W7_DIR = os.path.join(OUTPUTS_DIR, "week7_outputs")
W8_DIR = os.path.join(OUTPUTS_DIR, "week8_outputs")
W9_DIR = os.path.join(OUTPUTS_DIR, "week9_outputs")


def run_week1():
    print("\n" + "=" * 60)
    print("WEEK 1: Fetching papers from arXiv")
    print("=" * 60)
    import fetch_papers
    fetch_papers.main()


def run_week2():
    print("\n" + "=" * 60)
    print("WEEK 2: Preprocessing dataset")
    print("=" * 60)
    import preprocess
    raw_path    = os.path.join(DATA_DIR, "raw_papers.json")
    output_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    report_path = os.path.join(DATA_DIR, "dataset_stats.txt")
    cleaned = preprocess.preprocess(raw_path, output_path)
    preprocess.generate_stats_report(cleaned, report_path)


def run_week3():
    print("\n" + "=" * 60)
    print("WEEK 3: Baseline Transformer Summarization + ROUGE")
    print("=" * 60)
    import baseline_summarization
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path  = os.path.join(OUTPUTS_DIR, "baseline_summaries.json")
    baseline_summarization.run_baseline_evaluation(cleaned_path, output_path, n_papers=10)


def run_week4():
    print("\n" + "=" * 60)
    print("WEEK 4: Clustering + Joint Summarization")
    print("=" * 60)
    import clustering_summarization
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path  = os.path.join(OUTPUTS_DIR, "clustering_results.json")
    clustering_summarization.run_clustering_evaluation(cleaned_path, output_path, n_clusters=4)


def run_week5():
    print("\n" + "=" * 60)
    print("WEEK 5: SciSpacy NER Pipeline")
    print("=" * 60)
    import ner_pipeline
    cleaned_path    = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path     = os.path.join(OUTPUTS_DIR, "ner_results.json")
    global_freq_path = os.path.join(OUTPUTS_DIR, "entity_frequencies.json")
    ner_pipeline.run_ner_pipeline(cleaned_path, output_path, global_freq_path, n_papers=15)


def run_week6():
    print("\n" + "=" * 60)
    print("WEEK 6: Entity Normalization & Deduplication")
    print("=" * 60)
    import entity_normalization
    os.makedirs(W6_DIR, exist_ok=True)
    ner_path    = os.path.join(OUTPUTS_DIR, "ner_results.json")
    output_path = os.path.join(W6_DIR, "canonical_entities.json")
    entity_normalization.run_normalization(ner_path, output_path)


def run_week7():
    print("\n" + "=" * 60)
    print("WEEK 7: Initial NetworkX Knowledge Graph")
    print("=" * 60)
    import networkx_graph
    os.makedirs(W7_DIR, exist_ok=True)
    entities_path = os.path.join(W6_DIR, "canonical_entities.json")
    graph_pkl     = os.path.join(W7_DIR, "knowledge_graph.pkl")
    metrics_path  = os.path.join(W7_DIR, "metrics_report.json")
    viz_path      = os.path.join(W7_DIR, "week7_graph.png")
    networkx_graph.run_graph_construction(entities_path, graph_pkl, metrics_path, viz_path)


def run_week8():
    print("\n" + "=" * 60)
    print("WEEK 8: Enhanced Edge Construction")
    print("=" * 60)
    import edge_construction
    os.makedirs(W8_DIR, exist_ok=True)
    entities_path   = os.path.join(W6_DIR, "canonical_entities.json")
    cleaned_path    = os.path.join(DATA_DIR, "cleaned_dataset.json")
    graph_json      = os.path.join(W8_DIR, "enhanced_graph.json")
    graph_graphml   = os.path.join(W8_DIR, "enhanced_graph.graphml")
    metrics_csv     = os.path.join(W8_DIR, "metrics_report.csv")
    viz_path        = os.path.join(W8_DIR, "week8_graph.png")
    edge_construction.run_week8_pipeline(
        entities_path, cleaned_path,
        graph_json, graph_graphml, metrics_csv, viz_path,
    )


def run_week9():
    print("\n" + "=" * 60)
    print("WEEK 9: Community Detection & Graph Analytics")
    print("=" * 60)
    import community_detection
    os.makedirs(W9_DIR, exist_ok=True)
    graph_json = os.path.join(W8_DIR, "enhanced_graph.json")
    community_detection.INPUT_PATH  = graph_json
    community_detection.OUTPUT_DIR  = W9_DIR
    community_detection.main()


WEEK_RUNNERS = {
    1: run_week1,
    2: run_week2,
    3: run_week3,
    4: run_week4,
    5: run_week5,
    6: run_week6,
    7: run_week7,
    8: run_week8,
    9: run_week9,
}

WEEK_DESCRIPTIONS = {
    1: "Fetch papers from arXiv",
    2: "Preprocess dataset",
    3: "Baseline summarization + ROUGE",
    4: "Clustering + joint summarization",
    5: "SciSpacy NER pipeline",
    6: "Entity normalization & deduplication",
    7: "Initial NetworkX knowledge graph",
    8: "Enhanced edge construction (dep + PMI)",
    9: "Community detection & graph analytics",
}


def main():
    parser = argparse.ArgumentParser(
        description="Research Pipeline: Weeks 1–9",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="\n".join(
            f"  Week {w}: {d}" for w, d in WEEK_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "--weeks", nargs="+", type=int, default=list(range(1, 10)),
        help="Which weeks to run (default: all 1–9)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available weeks and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available weeks:")
        for w, d in WEEK_DESCRIPTIONS.items():
            print(f"  {w}: {d}")
        return

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    selected = sorted(set(args.weeks))
    unknown  = [w for w in selected if w not in WEEK_RUNNERS]
    if unknown:
        print(f"[WARN] Unknown week(s) ignored: {unknown}")

    for week_num in selected:
        if week_num not in WEEK_RUNNERS:
            continue
        try:
            WEEK_RUNNERS[week_num]()
        except Exception as exc:
            print(f"\n[ERROR] Week {week_num} failed: {exc}")
            raise

    print("\n All selected weeks complete.")
    print(f"   Data    → {DATA_DIR}/")
    print(f"   Outputs → {OUTPUTS_DIR}/")
    if 9 in selected:
        print(f"   W9      → {W9_DIR}/")


if __name__ == "__main__":
    main()
 