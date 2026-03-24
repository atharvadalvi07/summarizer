import json
import os
import time
from dataclasses import dataclass, asdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@dataclass
class SummaryResult:
    paper_id: str
    title: str
    original: str
    summary: str
    word_count_original: int
    word_count_summary: int
    compression_ratio: float
    rouge1: float
    rouge2: float
    rougeL: float


_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        print("Loading facebook/bart-large-cnn (first run downloads ~1.6 GB)...")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,
            framework="pt",
        )
        print("✓ Model loaded")
    return _summarizer


_rouge = None

def get_rouge():
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer
        _rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _rouge


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    scorer = get_rouge()
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def summarize_text(text: str, max_len: int = 130, min_len: int = 30) -> str:
    summarizer = get_summarizer()
    words = text.split()
    if len(words) > 900:
        text = " ".join(words[:900])
    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]


def compression_ratio(original: str, summary: str) -> float:
    orig_words = len(original.split())
    summ_words = len(summary.split())
    if orig_words == 0:
        return 0.0
    return round(summ_words / orig_words, 4)


def run_baseline_evaluation(
    cleaned_path: str,
    output_path: str,
    n_papers: int = 10,
) -> list[SummaryResult]:
    with open(cleaned_path) as f:
        papers = json.load(f)

    if n_papers:
        papers = papers[:n_papers]

    print(f"\nRunning baseline summarisation on {len(papers)} papers...")
    results: list[SummaryResult] = []

    for i, paper in enumerate(papers, 1):
        text = paper["abstract_clean"]
        if len(text.split()) < 30:
            print(f"  [{i}/{len(papers)}] Skipping (too short): {paper['title'][:60]}")
            continue

        print(f"  [{i}/{len(papers)}] {paper['title'][:60]}...")
        t0 = time.time()
        summary = summarize_text(text)
        elapsed = time.time() - t0

        rouge = compute_rouge(text, summary)
        cr = compression_ratio(text, summary)

        res = SummaryResult(
            paper_id=paper["id"],
            title=paper["title"],
            original=text,
            summary=summary,
            word_count_original=len(text.split()),
            word_count_summary=len(summary.split()),
            compression_ratio=cr,
            rouge1=rouge["rouge1"],
            rouge2=rouge["rouge2"],
            rougeL=rouge["rougeL"],
        )
        results.append(res)
        print(f"         ROUGE-L={res.rougeL:.3f}  CR={cr:.3f}  ({elapsed:.1f}s)")

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n✓ Results saved → {output_path}")

    print_summary_table(results)
    return results


def print_summary_table(results: list[SummaryResult]):
    if not results:
        return
    print("\n" + "="*70)
    print(f"{'Title':<40} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'CR':>6}")
    print("-"*70)
    for r in results:
        print(f"{r.title[:39]:<40} {r.rouge1:>6.3f} {r.rouge2:>6.3f} {r.rougeL:>6.3f} {r.compression_ratio:>6.3f}")
    print("="*70)
    avg_r1 = sum(r.rouge1 for r in results) / len(results)
    avg_r2 = sum(r.rouge2 for r in results) / len(results)
    avg_rl = sum(r.rougeL for r in results) / len(results)
    avg_cr = sum(r.compression_ratio for r in results) / len(results)
    print(f"{'AVERAGE':<40} {avg_r1:>6.3f} {avg_r2:>6.3f} {avg_rl:>6.3f} {avg_cr:>6.3f}")
    print("="*70)


if __name__ == "__main__":
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path = os.path.join(OUTPUTS_DIR, "baseline_summaries.json")

    if not os.path.exists(cleaned_path):
        print("cleaned_dataset.json not found. Run week2/preprocess.py first.")
        exit(1)

    run_baseline_evaluation(cleaned_path, output_path, n_papers=10)