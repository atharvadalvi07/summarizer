import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
import time

ARXIV_API = "http://export.arxiv.org/api/query"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_arxiv_papers(query: str, max_results: int = 20) -> list[dict]:
    """Fetch papers from arXiv given a search query."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    print(f"  Fetching: {url[:80]}...")

    with urllib.request.urlopen(url) as response:
        raw = response.read()

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(raw)
    papers = []

    for entry in root.findall("atom:entry", ns):
        paper = {
            "id": entry.find("atom:id", ns).text.strip(),
            "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
            "abstract": entry.find("atom:summary", ns).text.strip().replace("\n", " "),
            "authors": [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
            ],
            "published": entry.find("atom:published", ns).text,
            "categories": [
                c.attrib.get("term", "")
                for c in entry.findall("atom:category", ns)
            ],
        }
        papers.append(paper)

    return papers


def main():
    queries = [
        "multi-document summarization transformer",
        "knowledge graph construction NLP",
        "named entity recognition scientific text",
    ]

    all_papers = {}
    for q in queries:
        print(f"\nQuery: '{q}'")
        papers = fetch_arxiv_papers(q, max_results=15)
        all_papers[q] = papers
        print(f"  → {len(papers)} papers fetched")
        time.sleep(3)  # Be polite to arXiv API

    out_path = os.path.join(DATA_DIR, "raw_papers.json")
    with open(out_path, "w") as f:
        json.dump(all_papers, f, indent=2)

    total = sum(len(v) for v in all_papers.values())
    print(f"\n✓ Saved {total} papers to {out_path}")
    return all_papers


if __name__ == "__main__":
    main()