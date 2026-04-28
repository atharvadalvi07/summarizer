import json
import re
import csv
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process


TEXT_KEYS      = ["text", "entity", "word", "surface_form", "span", "mention", "value"]
LABEL_KEYS     = ["label", "entity_type", "type", "ent_type", "category", "tag"]
SOURCE_DOC_KEYS = ["source_doc", "doc_id", "paper_id", "document", "source", "id"]


def _resolve_key(d: dict, candidates: list[str], fallback=None):
    """Return the value of the first candidate key found in d."""
    for k in candidates:
        if k in d:
            return d[k]
    return fallback


def normalize_schema(raw_entities: list) -> list[dict]:
    """
    Accept Week 5 output in any of these shapes and normalise to a flat list
    of dicts with guaranteed keys: text, label, source_doc.

    Supported shapes
    ────────────────
    A) Flat list of entity dicts  (most common)
       [{"text": "BERT", "label": "METHOD", ...}, ...]

    B) Dict keyed by doc id
       {"paper_1": [{"entity": "BERT", ...}], "paper_2": [...]}

    C) List of paper dicts, each with an 'entities' sub-list
       [{"doc_id": "paper_1", "entities": [{"word": "BERT", ...}]}, ...]
    """
    normalised = []

    if isinstance(raw_entities, dict):
        for doc_id, ents in raw_entities.items():
            if not isinstance(ents, list):
                continue
            for e in ents:
                normalised.append(_to_canonical(e, fallback_source=doc_id))
        return normalised

    if (
        isinstance(raw_entities, list)
        and raw_entities
        and isinstance(raw_entities[0], dict)
        and any(k in raw_entities[0] for k in ("entities", "ents", "named_entities"))
    ):
        for paper in raw_entities:
            doc_id = _resolve_key(paper, SOURCE_DOC_KEYS, fallback="unknown")
            sub_key = next((k for k in ("entities", "ents", "named_entities") if k in paper), None)
            if sub_key is None:
                continue
            for e in paper[sub_key]:
                normalised.append(_to_canonical(e, fallback_source=doc_id))
        return normalised

    for e in raw_entities:
        normalised.append(_to_canonical(e))
    return normalised


def _to_canonical(e: dict, fallback_source: str = "unknown") -> dict:
    """Convert a single raw entity dict to the canonical schema."""
    text = _resolve_key(e, TEXT_KEYS, fallback="")
    if not isinstance(text, str):
        text = str(text)

    label      = _resolve_key(e, LABEL_KEYS,      fallback="UNKNOWN")
    source_doc = _resolve_key(e, SOURCE_DOC_KEYS, fallback=fallback_source)

    canonical = {k: v for k, v in e.items()}
    canonical["text"]       = text
    canonical["label"]      = label
    canonical["source_doc"] = source_doc
    return canonical




def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(
        r"\b(bert|gpt|llm|nlp|ner|kg)\b",
        lambda m: m.group().upper(),
        text,
    )
    return text


ABBREVIATION_MAP = {
    "bert": "bidirectional encoder representations from transformers",
    "gpt":  "generative pre-trained transformer",
    "llm":  "large language model",
    "nlp":  "natural language processing",
    "ner":  "named entity recognition",
    "kg":   "knowledge graph",
    "ml":   "machine learning",
    "dl":   "deep learning",
    "cv":   "computer vision",
    "qa":   "question answering",
}


def expand_abbreviations(text: str, abbrev_map: dict) -> str:
    words = text.split()
    return " ".join(abbrev_map.get(w.lower(), w) for w in words)


def deduplicate_entities(
    entities: list[dict],
    similarity_threshold: int = 88,
) -> tuple[list[dict], dict[str, str]]:
    by_label = defaultdict(list)
    for ent in entities:
        by_label[ent["label"]].append(ent)

    canonical_entities: list[dict] = []
    merge_map: dict[str, str] = {}

    for label, group in by_label.items():
        seen_canonical: list[str] = []

        for ent in group:
            norm = ent["normalized_text"]

            if not seen_canonical:
                ent["variants"]  = [ent["text"]]
                ent["frequency"] = 1
                seen_canonical.append(norm)
                canonical_entities.append(ent)
                merge_map[norm] = norm
                continue

            result = process.extractOne(
                norm,
                seen_canonical,
                scorer=fuzz.token_sort_ratio,
            )

            if result is None:
                match, score, idx = norm, 0, -1
            else:
                match, score, idx = result

            if score >= similarity_threshold:
                canonical_idx = next(
                    (
                        i for i, ce in enumerate(canonical_entities)
                        if ce["label"] == label
                        and ce["normalized_text"] == match
                    ),
                    None,
                )
                if canonical_idx is not None:
                    canonical_entities[canonical_idx]["variants"].append(ent["text"])
                    canonical_entities[canonical_idx]["frequency"] = (
                        canonical_entities[canonical_idx].get("frequency", 1) + 1
                    )
                merge_map[norm] = match
            else:
                ent["variants"]  = [ent["text"]]
                ent["frequency"] = 1
                seen_canonical.append(norm)
                canonical_entities.append(ent)
                merge_map[norm] = norm

    return canonical_entities, merge_map


def load_entities(filepath: str) -> list:
    with open(filepath, "r") as f:
        return json.load(f)


def run_week6_pipeline(
    input_path: str,
    output_json: str   = "week6_cleaned_entities.json",
    output_csv: str    = "week6_cleaned_entities.csv",
    merge_map_path: str = "week6_merge_map.json",
    use_umls: bool     = False,
    fuzzy_threshold: int = 88,
):
    print("=" * 60)
    print("Week 6 — Entity Normalization & Deduplication")
    print("=" * 60)

    raw = load_entities(input_path)
    print(f"[1/5] Loaded raw data from '{input_path}'")
    entities = normalize_schema(raw)
    # drop blanks
    entities = [e for e in entities if e.get("text", "").strip()]
    print(f"[2/5] Schema resolved → {len(entities)} entities "
          f"(keys: text / label / source_doc guaranteed)")
    
    for ent in entities:
        norm = normalize_text(ent["text"])
        norm = expand_abbreviations(norm, ABBREVIATION_MAP)
        ent["normalized_text"] = norm
    print(f"[3/5] Surface-form normalisation complete")
    
    cleaned, merge_map = deduplicate_entities(entities, fuzzy_threshold)
    n_removed = len(entities) - len(cleaned)
    print(f"[4/5] Deduplication — {len(cleaned)} canonical entities "
          f"({n_removed} duplicates merged, threshold={fuzzy_threshold})")

    if use_umls:
        print("[5/5] UMLS linking enabled but library stubs are commented out; skipping.")
    else:
        print("[5/5] UMLS linking skipped (set use_umls=True to enable)")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(cleaned, f, indent=2)

    with open(merge_map_path, "w") as f:
        json.dump(merge_map, f, indent=2)

    fieldnames = ["normalized_text", "label", "frequency", "variants",
                  "source_doc", "umls_cui", "umls_name"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ent in cleaned:
            row = dict(ent)
            row["variants"] = " | ".join(ent.get("variants", []))
            writer.writerow(row)

    print(f"\n✓ Outputs saved:")
    print(f"  • {output_json}")
    print(f"  • {output_csv}")
    print(f"  • {merge_map_path}")
    print("Done ✓")
    return cleaned, merge_map

#for run_all.py
def run_normalization(input_path: str, output_path: str):
    """Thin wrapper for run_all.py integration."""
    stem        = str(Path(output_path).with_suffix(""))
    output_csv  = stem + ".csv"
    merge_map   = stem + "_merge_map.json"
    return run_week6_pipeline(
        input_path=input_path,
        output_json=output_path,
        output_csv=output_csv,
        merge_map_path=merge_map,
    )


def print_entity_stats(entities: list[dict]):
    label_counts = defaultdict(int)
    for ent in entities:
        label_counts[ent["label"]] += 1
    print("\n── Entity Type Breakdown ──────────────────")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {count:>5}")
    print(f"  {'TOTAL':<20} {sum(label_counts.values()):>5}")


# Standalone run
if __name__ == "__main__":
    cleaned_entities, merge_map = run_week6_pipeline(
        input_path="week5_entities.json",
        output_json="week6_cleaned_entities.json",
        output_csv="week6_cleaned_entities.csv",
        merge_map_path="week6_merge_map.json",
        use_umls=False,
        fuzzy_threshold=88,
    )
    print_entity_stats(cleaned_entities)



from scispacy.linking import EntityLinker

def link_to_umls(entities, nlp):
    try:
        linker = nlp.get_pipe("scispacy_linker")
    except KeyError:
        print("[INFO] UMLS linker not loaded; skipping.")
        return entities
    for ent in entities:
        doc = nlp(ent["normalized_text"])
        for span in doc.ents:
            if span._.kb_ents:
                top_cui, score = span._.kb_ents[0]
                concept = linker.kb.cui_to_entity[top_cui]
                ent["umls_cui"]   = top_cui
                ent["umls_name"]  = concept.canonical_name
                ent["umls_score"] = round(score, 3)
                break
    return entities

