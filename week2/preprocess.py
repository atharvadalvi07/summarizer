import json
import os
import re
import string
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text: str, keep_punct: bool = True) -> str:
    text = text.encode("ascii", "ignore").decode()
    if not keep_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def sentence_segment(text: str) -> list[str]:
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def clean_abstract(abstract: str) -> str:
    text = normalize_whitespace(abstract)
    text = remove_special_chars(text, keep_punct=True)
    text = re.sub(r"\$[^$]+\$", "[MATH]", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    return text.strip()


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    seen_ids = set()
    unique = []
    for p in papers:
        pid = p["id"].split("/")[-1]
        base_id = re.sub(r"v\d+$", "", pid)
        if base_id not in seen_ids:
            seen_ids.add(base_id)
            unique.append(p)
    return unique


def preprocess(raw_path: str, output_path: str) -> list[dict]:
    with open(raw_path) as f:
        raw = json.load(f)

    all_papers: list[dict] = []
    for papers in raw.values():
        all_papers.extend(papers)

    print(f"Raw total   : {len(all_papers)} papers")
    papers = deduplicate_papers(all_papers)
    print(f"After dedup : {len(papers)} papers")

    cleaned = []
    for p in papers:
        cp = {
            "id": p["id"],
            "title": normalize_whitespace(p["title"]),
            "abstract_raw": p["abstract"],
            "abstract_clean": clean_abstract(p["abstract"]),
            "sentences": sentence_segment(clean_abstract(p["abstract"])),
            "authors": p["authors"],
            "published": p["published"],
            "categories": p["categories"],
            "word_count": len(clean_abstract(p["abstract"]).split()),
        }
        cleaned.append(cp)

    wc = [p["word_count"] for p in cleaned]
    print(f"Word count  : min={min(wc)}  max={max(wc)}  avg={sum(wc)//len(wc)}")

    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f" Saved cleaned dataset → {output_path}")
    return cleaned


def generate_stats_report(cleaned: list[dict], report_path: str):
    cat_counter: dict = defaultdict(int)
    for p in cleaned:
        for c in p["categories"]:
            cat_counter[c] += 1

    top_cats = sorted(cat_counter.items(), key=lambda x: -x[1])[:10]

    with open(report_path, "w") as f:
        f.write("=== Dataset Statistics Report (Week 2) ===\n\n")
        f.write(f"Total papers       : {len(cleaned)}\n")
        f.write(f"Total sentences    : {sum(len(p['sentences']) for p in cleaned)}\n")
        avg_wc = sum(p['word_count'] for p in cleaned) // len(cleaned)
        f.write(f"Avg words/abstract : {avg_wc}\n\n")
        f.write("Top arXiv categories:\n")
        for cat, count in top_cats:
            f.write(f"  {cat:<20} {count}\n")

    print(f" Stats report saved → {report_path}")


if __name__ == "__main__":
    raw_path = os.path.join(DATA_DIR, "raw_papers.json")

    if not os.path.exists(raw_path):
        print("raw_papers.json not found — running week1/fetch_papers.py first...")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
        import fetch_papers
        fetch_papers.main()

    output_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    report_path = os.path.join(DATA_DIR, "dataset_stats.txt")

    cleaned = preprocess(raw_path, output_path)
    generate_stats_report(cleaned, report_path)