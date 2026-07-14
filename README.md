# Multi-Document Summarization and Knowledge Graph Construction for Enhanced Research Comprehension

A research pipeline that combines transformer-based multi-document summarization with automated knowledge graph construction to support structured comprehension across a corpus of scientific papers. Built as a semester-long research project at Texas Tech University.

## Overview

The pipeline processes a corpus of 45 open-access arXiv papers spanning named entity recognition, multi-document summarization, NLP education, and knowledge graph construction. It runs each paper through baseline and cluster-based summarization, domain-aware named entity recognition, entity normalization, knowledge graph construction (co-occurrence and dependency-based edges), community detection, and interactive visualization, then evaluates the results with both quantitative metrics (ROUGE, BLEU, graph structural metrics) and structured qualitative assessment.

Full methodology and results are documented in the final manuscript.

## Pipeline Structure

| Week | Stage | Script | Description |
|------|-------|--------|-------------|
| 1 | Data Acquisition | `week1/fetch_papers.py` | Fetches papers from the arXiv API |
| 2 | Preprocessing | `week2/preprocess.py` | Cleans, deduplicates, and sentence-segments abstracts |
| 3 | Baseline Summarization | `week3/baseline_summarization.py` | Per-document summarization with `facebook/bart-large-cnn` + ROUGE evaluation |
| 4 | Cluster-Based Summarization | `week4/clustering_summarization.py` | TF-IDF/K-Means topic clustering, individual-merge vs. joint summarization |
| 5 | Named Entity Recognition | `week5/ner_pipeline.py` | SciSpacy/spaCy entity extraction and categorization |
| 6 | Entity Normalization | `week6/entity_normalization.py` | Fuzzy deduplication (RapidFuzz) and canonicalization |
| 7 | Knowledge Graph Construction | `week7/networkx_graph.py` | Initial co-occurrence graph in NetworkX |
| 8 | Enhanced Edge Construction | `week8/edge_construction.py` | Adds PMI-normalized co-occurrence weights and dependency-based (SVO) edges |
| 9 | Community Detection | `week9/community_detection.py` | Louvain, greedy modularity, and Girvan-Newman partitioning |
| 10 | Visualization | `week10/visualization.py` | Interactive PyVis HTML graph |
| 11 | Filtering & Layer Controls | `week11/customize.py` | Custom JS canvas renderer with community/entity/edge-type filtering |
| 12 | Evaluation | `week12/eval.py` | Consolidated summarization and graph-structural evaluation report |

`run_all.py` orchestrates the full pipeline end to end, with per-week output subdirectories and per-stage error handling.

## Setup

```bash
pip install -r requirements.txt
```

SciSpacy's scientific NER model is not on PyPI and must be installed separately:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

Requires Python 3.10+ and, for reasonable summarization runtime, a machine with a GPU (the pipeline defaults to CPU inference if none is available).

## Usage

Run the full pipeline:

```bash
python run_all.py
```

Run specific weeks:

```bash
python run_all.py --weeks 1 2 3
```

List available weeks:

```bash
python run_all.py --list
```

## Outputs

Running the pipeline generates a `data/` directory (raw and cleaned datasets) and an `outputs/` directory (per-week results, graphs, metrics, and visualizations). Neither is checked into this repository; both are produced locally on each run.

## Results Summary

- **Summarization:** Joint cluster-based summarization outperformed individual-merge summarization on all ROUGE metrics (ROUGE-1: 0.2299 vs. 0.1599).
- **Knowledge graph:** 301 entities, 8,640 edges after enhanced (co-occurrence + dependency) construction, average clustering coefficient 0.79.
- **Community detection:** Louvain partitioning identified 7 communities (modularity 0.18).
- **Qualitative evaluation:** System-assisted reading reduced time-to-first-understanding by approximately 64% relative to raw document review.

Full results, methodology, and discussion are in the final manuscript.

## Repository Structure

```
summarizer/
├── run_all.py
├── requirements.txt
├── week1/   fetch_papers.py
├── week2/   preprocess.py
├── week3/   baseline_summarization.py
├── week4/   clustering_summarization.py
├── week5/   ner_pipeline.py
├── week6/   entity_normalization.py
├── week7/   networkx_graph.py
├── week8/   edge_construction.py
├── week9/   community_detection.py
├── week10/  visualization.py
├── week11/  customize.py
└── week12/  eval.py
```

## Author

Atharva Dalvi
B.S. Computer Science, Texas Tech University

## Acknowledgments

This project was completed as a supervised semester-long research project. Thank you to my supervising professor for the guidance and support throughout.
