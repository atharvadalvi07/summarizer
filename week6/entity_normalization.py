import json
import re
import csv
from collections import defaultdict
from pathlib import Path


from rapidfuzz import fuzz, process


def load_entities(filepath: str) -> list[dict]:
    """Load extracted entities from Week 5 JSON output."""
    with open(filepath, "r") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    Normalize entity surface form:
    - Lowercase
    - Strip extra whitespace
    - Remove punctuation artifacts
    - Expand common abbreviations
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)              
    text = re.sub(r"[^\w\s\-]", "", text)          # remove punctuation 
    text = re.sub(r"\b(bert|gpt|llm|nlp|ner|kg)\b",
                  lambda m: m.group().upper(), text)  # keep known acronyms uppercase
    return text


ABBREVIATION_MAP = {
    "bert": "bidirectional encoder representations from transformers",
    "gpt": "generative pre-trained transformer",
    "llm": "large language model",
    "nlp": "natural language processing",
    "ner": "named entity recognition",
    "kg": "knowledge graph",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "qa": "question answering",
}


def expand_abbreviations(text: str, abbrev_map: dict) -> str:
    """Expand known abbreviations to their full forms."""
    words = text.split()
    expanded = [abbrev_map.get(w.lower(), w) for w in words]
    return " ".join(expanded)


def deduplicate_entities(
    entities: list[dict],
    similarity_threshold: int = 88
) -> tuple[list[dict], dict[str, str]]:
    """
    Deduplicate entities using fuzzy string matching.

    Args:
        entities: List of entity dicts with 'text', 'label', 'source_doc', etc.
        similarity_threshold: RapidFuzz score (0–100) above which two entities
                              are considered duplicates.

    Returns:
        (canonical_entities, merge_map)
        - canonical_entities: deduplicated list, each with a 'variants' field
        - merge_map: {variant_text -> canonical_text}
    """

    by_label = defaultdict(list)
    for ent in entities:
        by_label[ent["label"]].append(ent)

    canonical_entities = []
    merge_map = {}  # variant -> canonical

    for label, group in by_label.items():
        seen_canonical = []  

        for ent in group:
            norm = ent["normalized_text"]

            if not seen_canonical:
                ent["variants"] = [ent["text"]]
                seen_canonical.append(norm)
                canonical_entities.append(ent)
                merge_map[norm] = norm
                continue

            match, score, idx = process.extractOne(
                norm,
                seen_canonical,
                scorer=fuzz.token_sort_ratio
            )

            if score >= similarity_threshold:
                canonical_entities[idx]["variants"].append(ent["text"])
                canonical_entities[idx]["frequency"] = (
                    canonical_entities[idx].get("frequency", 1) + 1
                )
                merge_map[norm] = match
            else:
                ent["variants"] = [ent["text"]]
                ent["frequency"] = 1
                seen_canonical.append(norm)
                canonical_entities.append(ent)
                merge_map[norm] = norm

    return canonical_entities, merge_map




def run_week6_pipeline(
    input_path: str,
    output_json: str = "week6_cleaned_entities.json",
    output_csv: str = "week6_cleaned_entities.csv",
    merge_map_path: str = "week6_merge_map.json",
    use_umls: bool = False,
    fuzzy_threshold: int = 88,
):
    print("=" * 60)
    print("Week 6 — Entity Normalization & Deduplication")
    print("=" * 60)

    # Load ──────────────────────────────────────────────
    raw_entities = load_entities(input_path)
    print(f"[1/5] Loaded {len(raw_entities)} raw entities from '{input_path}'")

    # Normalize text ────────────────────────────────────
    for ent in raw_entities:
        norm = normalize_text(ent["text"])
        norm = expand_abbreviations(norm, ABBREVIATION_MAP)
        ent["normalized_text"] = norm

    print(f"[2/5] Normalized all entity surface forms")

    # Deduplicate ───────────────────────────────────────
    cleaned, merge_map = deduplicate_entities(raw_entities, fuzzy_threshold)
    n_removed = len(raw_entities) - len(cleaned)
    print(f"[3/5] Deduplication complete — {len(cleaned)} canonical entities "
          f"({n_removed} duplicates merged, threshold={fuzzy_threshold})")


    # Save outputs ──────────────────────────────────────
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

    print(f"[5/5] Saved outputs:")
    print(f"      • {output_json}  — cleaned entity list (JSON)")
    print(f"      • {output_csv}   — human-readable CSV")
    print(f"      • {merge_map_path} — variant → canonical map")
    print("Done ✓")
    return cleaned, merge_map

def print_entity_stats(entities: list[dict]):
    """Print a summary breakdown by entity label."""
    label_counts = defaultdict(int)
    for ent in entities:
        label_counts[ent["label"]] += 1

    print("\n── Entity Type Breakdown ──────────────────")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {count:>5}")
    print(f"  {'TOTAL':<20} {sum(label_counts.values()):>5}")


if __name__ == "__main__":
    # Expected input: JSON list 
    
    cleaned_entities, merge_map = run_week6_pipeline(
        input_path="week5_entities.json",   # output Week 5
        output_json="week6_cleaned_entities.json",
        output_csv="week6_cleaned_entities.csv",
        merge_map_path="week6_merge_map.json",
        use_umls=False,       
        fuzzy_threshold=88,
    )

    print_entity_stats(cleaned_entities)















# import spacy
# import scispacy
# from scispacy.linking import EntityLinker
# def link_to_umls(entities: list[dict], nlp) -> list[dict]:
#     """
#     Use SciSpacy's EntityLinker to map entities to UMLS concept IDs.
#     Requires: pip install scispacy && python -m spacy download en_core_sci_sm
#               pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
#     """
#     try:
#         linker = nlp.get_pipe("scispacy_linker")
#     except KeyError:
#         print("[INFO] UMLS linker not loaded; skipping concept linking.")
#         return entities

#     for ent in entities:
#         doc = nlp(ent["normalized_text"])
#         for span in doc.ents:
#             if span._.kb_ents:
#                 top_cui, score = span._.kb_ents[0]
#                 concept = linker.kb.cui_to_entity[top_cui]
#                 ent["umls_cui"] = top_cui
#                 ent["umls_name"] = concept.canonical_name
#                 ent["umls_score"] = round(score, 3)
#                 break

#     return entities
# # Optional UMLS linking ────────────────────────────
#     if use_umls:
#         print("[4/5] Loading SciSpacy model for UMLS linking …")
#         nlp = spacy.load("en_core_sci_sm")
#         nlp.add_pipe("scispacy_linker",
#                      config={"resolve_abbreviations": True, "linker_name": "umls"})
#         cleaned = link_to_umls(cleaned, nlp)
#         print("[4/5] UMLS concept linking done")
#     else:
#         print("[4/5] UMLS linking skipped (set use_umls=True to enable)")