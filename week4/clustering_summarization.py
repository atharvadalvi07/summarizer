import json
import os
import re
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_summarizer():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1,
        framework="pt",
    )


def get_rouge():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def cluster_documents(texts: list[str], n_clusters: int = 4):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels, metric="cosine", sample_size=min(500, len(texts)))
        print(f"  Silhouette score (k={n_clusters}): {sil:.3f}")

    return labels, vectorizer, X.toarray()


def get_cluster_top_terms(vectorizer, X, labels, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for cid in np.unique(labels):
        mask = labels == cid
        centroid = X[mask].mean(axis=0)
        top_idx = centroid.argsort()[::-1][:top_n]
        cluster_terms[int(cid)] = [feature_names[i] for i in top_idx]
    return cluster_terms


def truncate_to_words(text: str, max_words: int = 900) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def strategy_individual_merge(papers: list[dict], summarizer) -> str:
    individual_summaries = []
    for p in papers:
        text = truncate_to_words(p["abstract_clean"])
        out = summarizer(text, max_length=80, min_length=20, do_sample=False, truncation=True)
        individual_summaries.append(out[0]["summary_text"])
    merged = truncate_to_words(" ".join(individual_summaries))
    final = summarizer(merged, max_length=150, min_length=40, do_sample=False, truncation=True)
    return final[0]["summary_text"]

def truncate_to_tokens(text: str, tokenizer, max_tokens: int = 1024):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def strategy_joint(papers: list[dict], summarizer) -> str:
    tokenizer = summarizer.tokenizer

    combined = " ".join(p["abstract_clean"] for p in papers)
    combined = truncate_to_tokens(combined, tokenizer, max_tokens=1024)

    out = summarizer(
        combined,
        max_length=150,
        min_length=40,
        do_sample=False,
        truncation=True
    )
    return out[0]["summary_text"]

# def strategy_joint(papers: list[dict], summarizer) -> str:
#     combined = truncate_to_words(" ".join(p["abstract_clean"] for p in papers), max_words=900)
#     out = summarizer(combined, max_length=150, min_length=40, do_sample=False, truncation=True)
#     return out[0]["summary_text"]


@dataclass
class ClusterResult:
    cluster_id: int
    top_terms: list[str]
    n_papers: int
    paper_titles: list[str]
    individual_merge_summary: str
    joint_summary: str
    individual_rouge1: float
    individual_rouge2: float
    individual_rougeL: float
    joint_rouge1: float
    joint_rouge2: float
    joint_rougeL: float


def compute_rouge(rouge_scorer_obj, reference: str, hypothesis: str) -> dict:
    scores = rouge_scorer_obj.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def run_clustering_evaluation(cleaned_path: str, output_path: str, n_clusters: int = 4):
    with open(cleaned_path) as f:
        papers = json.load(f)

    texts = [p["abstract_clean"] for p in papers]
    print(f"Clustering {len(texts)} papers into {n_clusters} groups...")

    labels, vectorizer, X = cluster_documents(texts, n_clusters=n_clusters)
    cluster_terms = get_cluster_top_terms(vectorizer, X, labels)

    for cid in np.unique(labels):
        print(f"  Cluster {cid}: {int((labels==cid).sum())} papers | top terms: {cluster_terms[int(cid)]}")

    print("\nLoading summarization model...")
    summarizer = get_summarizer()
    rouge = get_rouge()

    results: list[ClusterResult] = []

    for cid in np.unique(labels):
        mask = labels == cid
        cluster_papers = [papers[i] for i, m in enumerate(mask) if m]
        print(f"\n--- Cluster {cid} ({len(cluster_papers)} papers) ---")

        # reference = truncate_to_words(" ".join(p["abstract_clean"] for p in cluster_papers), 1000)
        tokenizer = summarizer.tokenizer
        
        reference = truncate_to_tokens(
            " ".join(p["abstract_clean"] for p in cluster_papers),
            tokenizer,
            max_tokens=512)

        print("  Running individual-merge strategy...")
        ind_summary = strategy_individual_merge(cluster_papers, summarizer)

        print("  Running joint strategy...")
        joint_summary = strategy_joint(cluster_papers, summarizer)

        ind_rouge = compute_rouge(rouge, reference, ind_summary)
        joint_rouge = compute_rouge(rouge, reference, joint_summary)

        print(f"  Individual-Merge ROUGE-L : {ind_rouge['rougeL']:.3f}")
        print(f"  Joint           ROUGE-L : {joint_rouge['rougeL']:.3f}")

        results.append(ClusterResult(
            cluster_id=int(cid),
            top_terms=cluster_terms[int(cid)],
            n_papers=len(cluster_papers),
            paper_titles=[p["title"] for p in cluster_papers],
            individual_merge_summary=ind_summary,
            joint_summary=joint_summary,
            individual_rouge1=ind_rouge["rouge1"],
            individual_rouge2=ind_rouge["rouge2"],
            individual_rougeL=ind_rouge["rougeL"],
            joint_rouge1=joint_rouge["rouge1"],
            joint_rouge2=joint_rouge["rouge2"],
            joint_rougeL=joint_rouge["rougeL"],
        ))

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n✓ Results saved → {output_path}")

    print_comparison_table(results)
    return results


def print_comparison_table(results: list[ClusterResult]):
    print("\n" + "="*75)
    print(f"{'Cluster':<10} {'N':>4} {'Ind R-L':>8} {'Joint R-L':>10} {'Winner':>8}")
    print("-"*75)
    for r in results:
        winner = "Joint" if r.joint_rougeL >= r.individual_rougeL else "Indiv"
        print(f"Cluster {r.cluster_id:<3} {r.n_papers:>4} {r.individual_rougeL:>8.3f} {r.joint_rougeL:>10.3f} {winner:>8}")
    print("="*75)
    avg_ind = sum(r.individual_rougeL for r in results) / len(results)
    avg_jt = sum(r.joint_rougeL for r in results) / len(results)
    print(f"{'AVERAGE':<14} {avg_ind:>8.3f} {avg_jt:>10.3f}")


if __name__ == "__main__":
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path = os.path.join(OUTPUTS_DIR, "clustering_results.json")

    if not os.path.exists(cleaned_path):
        print("cleaned_dataset.json not found. Run week2/preprocess.py first.")
        exit(1)

    run_clustering_evaluation(cleaned_path, output_path, n_clusters=4)