import argparse
import sys
import os

ROOT = os.path.dirname(__file__)
for week in ["week1", "week2", "week3", "week4", "week5"]:
    sys.path.insert(0, os.path.join(ROOT, week))

DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")


def run_week1():
    print("\n" + "="*60)
    print("WEEK 1: Fetching papers from arXiv")
    print("="*60)
    import fetch_papers
    fetch_papers.main()


def run_week2():
    print("\n" + "="*60)
    print("WEEK 2: Preprocessing dataset")
    print("="*60)
    import preprocess
    raw_path = os.path.join(DATA_DIR, "raw_papers.json")
    output_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    report_path = os.path.join(DATA_DIR, "dataset_stats.txt")
    cleaned = preprocess.preprocess(raw_path, output_path)
    preprocess.generate_stats_report(cleaned, report_path)


def run_week3():
    print("\n" + "="*60)
    print("WEEK 3: Baseline Transformer Summarization + ROUGE")
    print("="*60)
    import baseline_summarization
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path = os.path.join(OUTPUTS_DIR, "baseline_summaries.json")
    baseline_summarization.run_baseline_evaluation(cleaned_path, output_path, n_papers=10)


def run_week4():
    print("\n" + "="*60)
    print("WEEK 4: Clustering + Joint Summarization")
    print("="*60)
    import clustering_summarization
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path = os.path.join(OUTPUTS_DIR, "clustering_results.json")
    clustering_summarization.run_clustering_evaluation(cleaned_path, output_path, n_clusters=4)


def run_week5():
    print("\n" + "="*60)
    print("WEEK 5: SciSpacy NER Pipeline")
    print("="*60)
    import ner_pipeline
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path = os.path.join(OUTPUTS_DIR, "ner_results.json")
    global_freq_path = os.path.join(OUTPUTS_DIR, "entity_frequencies.json")
    ner_pipeline.run_ner_pipeline(cleaned_path, output_path, global_freq_path, n_papers=15)


WEEK_RUNNERS = {
    1: run_week1,
    2: run_week2,
    3: run_week3,
    4: run_week4,
    5: run_week5,
}


def main():
    parser = argparse.ArgumentParser(description="Research Pipeline: Weeks 1–5")
    parser.add_argument(
        "--weeks", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Which weeks to run (default: all)"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    for week_num in sorted(args.weeks):
        if week_num not in WEEK_RUNNERS:
            print(f"Unknown week: {week_num}")
            continue
        WEEK_RUNNERS[week_num]()

    print("\n✅ All selected weeks complete.")
    print(f"   Data    → {DATA_DIR}/")
    print(f"   Outputs → {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()