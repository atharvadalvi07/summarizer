import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Method": [
        "bert", "gpt", "t5", "bart", "transformer", "lstm", "attention",
        "fine-tuning", "pre-training", "zero-shot", "few-shot", "rl",
        "reinforcement learning", "regression", "classification", "clustering",
        "k-means", "svm", "random forest", "neural network", "cnn", "rnn",
    ],
    "Dataset": [
        "dataset", "corpus", "benchmark", "arxiv", "pubmed", "squad",
        "cnn/dailymail", "xsum", "gigaword", "multi-news", "wikipedia",
        "conll", "ace", "ontonotes", "ms marco",
    ],
    "Task": [
        "summarization", "classification", "translation", "question answering",
        "named entity recognition", "ner", "relation extraction", "parsing",
        "coreference", "information extraction", "text generation",
        "knowledge graph", "text mining",
    ],
    "Technique": [
        "attention mechanism", "self-attention", "cross-attention",
        "beam search", "contrastive learning", "graph neural network",
        "knowledge distillation", "data augmentation", "dropout",
        "layer normalization", "positional encoding",
    ],
    "Institution": [
        "university", "institute", "laboratory", "lab", "google", "microsoft",
        "facebook", "meta", "openai", "deepmind", "allen institute",
        "stanford", "mit", "cmu", "oxford", "cambridge",
    ],
}


def heuristic_category(entity_text: str) -> str:
    lower = entity_text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "Other"


_nlp = None

def get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_sci_lg")
        print("✓ Loaded SciSpacy en_core_sci_lg")
    except OSError:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            print("⚠ SciSpacy not found; using en_core_web_sm")
        except OSError:
            raise RuntimeError("No spaCy model found. Run: python -m spacy download en_core_web_sm")
    return _nlp


@dataclass
class Entity:
    text: str
    label: str
    category: str
    start_char: int
    end_char: int
    source_paper_id: str


@dataclass
class PaperEntities:
    paper_id: str
    title: str
    entities: list[Entity] = field(default_factory=list)
    entity_count: int = 0
    categories: dict[str, int] = field(default_factory=dict)


def extract_entities_from_text(text: str, paper_id: str, nlp) -> list[Entity]:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if len(ent.text.strip()) <= 1:
            continue
        if re.fullmatch(r"[\d\W]+", ent.text):
            continue
        entities.append(Entity(
            text=ent.text.strip(),
            label=ent.label_,
            category=heuristic_category(ent.text),
            start_char=ent.start_char,
            end_char=ent.end_char,
            source_paper_id=paper_id,
        ))
    return entities


def normalize_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    seen: set[str] = set()
    unique = []
    for ent in entities:
        key = normalize_entity_text(ent.text)
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique


def run_ner_pipeline(cleaned_path, output_path, global_freq_path, n_papers=None):
    with open(cleaned_path) as f:
        papers = json.load(f)

    if n_papers:
        papers = papers[:n_papers]

    print(f"Running NER on {len(papers)} papers...")
    nlp = get_nlp()

    all_paper_entities = []
    global_entities = []

    for i, paper in enumerate(papers, 1):
        text = paper["abstract_clean"]
        pid = paper["id"]
        print(f"  [{i}/{len(papers)}] {paper['title'][:60]}...")

        raw_entities = extract_entities_from_text(text, pid, nlp)
        unique_entities = deduplicate_entities(raw_entities)
        category_counts = dict(Counter(e.category for e in unique_entities))

        pe = PaperEntities(
            paper_id=pid,
            title=paper["title"],
            entities=unique_entities,
            entity_count=len(unique_entities),
            categories=category_counts,
        )
        all_paper_entities.append(pe)
        global_entities.extend(unique_entities)

    serializable = [
        {
            "paper_id": pe.paper_id,
            "title": pe.title,
            "entity_count": pe.entity_count,
            "categories": pe.categories,
            "entities": [asdict(e) for e in pe.entities],
        }
        for pe in all_paper_entities
    ]

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✓ Per-paper entities saved → {output_path}")

    global_freq = Counter(normalize_entity_text(e.text) for e in global_entities)
    top_entities = [
        {"entity": ent, "count": cnt, "category": heuristic_category(ent)}
        for ent, cnt in global_freq.most_common(50)
    ]
    with open(global_freq_path, "w") as f:
        json.dump(top_entities, f, indent=2)
    print(f"✓ Global entity frequencies saved → {global_freq_path}")

    print_ner_summary(all_paper_entities, global_entities)
    return all_paper_entities


def print_ner_summary(paper_results, global_entities):
    print("\n" + "="*60)
    print("NER Pipeline Summary")
    print("="*60)
    total_ents = sum(p.entity_count for p in paper_results)
    print(f"Papers processed   : {len(paper_results)}")
    print(f"Total entities     : {total_ents}")
    print(f"Avg / paper        : {total_ents / max(len(paper_results), 1):.1f}")

    cat_totals: Counter = Counter()
    for p in paper_results:
        cat_totals.update(p.categories)

    print("\nEntities by category:")
    for cat, cnt in cat_totals.most_common():
        print(f"  {cat:<20} {cnt}")

    top10 = Counter(e.text.lower() for e in global_entities).most_common(10)
    print("\nTop 10 entities (global):")
    for ent, cnt in top10:
        print(f"  {ent:<30} {cnt}")
    print("="*60)


if __name__ == "__main__":
    cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.json")
    output_path = os.path.join(OUTPUTS_DIR, "ner_results.json")
    global_freq_path = os.path.join(OUTPUTS_DIR, "entity_frequencies.json")

    if not os.path.exists(cleaned_path):
        print("cleaned_dataset.json not found. Run week2/preprocess.py first.")
        exit(1)

    run_ner_pipeline(cleaned_path, output_path, global_freq_path, n_papers=15)