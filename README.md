# South Asian RAG Assistant

A Retrieval-Augmented Generation (RAG) question-answering system specialised in South Asian cuisine. The system retrieves relevant passages from a curated knowledge corpus and generates grounded, context-aware answers using a small language model.

Developed as part of **COMP64702: Transforming Text Into Meaning**.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [Corpus and Benchmark](#corpus-and-benchmark)
- [Retrieval Strategy](#retrieval-strategy)
- [Evaluation Results](#evaluation-results)
- [Domain Coverage](#domain-coverage)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

This project implements a domain-specific RAG pipeline for answering questions about South Asian cuisine. Given a natural language query, the system retrieves the most relevant passages from a 204-document corpus, constructs an adaptive prompt, and generates a grounded answer using `Qwen2.5-0.5B-Instruct`.

The system is evaluated against a manually curated 100-question benchmark spanning factual, procedural, comparative, ingredient, and technique questions across eight countries: India, Pakistan, Bangladesh, Sri Lanka, Nepal, Afghanistan, Bhutan, and the Maldives.

---

## System Architecture

```
Query
  │
  ▼
┌─────────────────────────────────┐
│         Hybrid Retrieval        │
│  ┌─────────────┐ ┌───────────┐  │
│  │ Dense (MiniLM│ │BM25Okapi │  │
│  │ all-MiniLM  │ │ stemmed  │  │
│  │  L6-v2)     │ │ tokens   │  │
│  └──────┬──────┘ └─────┬─────┘  │
│         └──────┬────────┘       │
│          RRF Merge & Re-rank    │
│              Top-5              │
└──────────────┬──────────────────┘
               │
               ▼
    ┌───────────────────────┐
    │   Adaptive Prompting  │
    │  (query-type routing) │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Qwen2.5-0.5B-Instruct│
    │     (generation)      │
    └───────────┬───────────┘
                │
                ▼
             Answer
```

**Ingestion** — Documents are sentence-boundary chunked (220-word max, 40-word overlap) and embedded using `all-MiniLM-L6-v2` with a title prefix. Short documents under 300 characters and chunks under 25 words are filtered out.

**Retrieval** — A hybrid approach combines dense cosine similarity and BM25Okapi (with Snowball stemming). Each method retrieves 40 candidates; these are merged and re-ranked using Reciprocal Rank Fusion (RRF, k=60), yielding the final top-5 chunks.

**Generation** — Retrieved chunks are assembled into a context window (max 1,800 characters) and passed to `Qwen2.5-0.5B-Instruct` via an adaptive prompt that adjusts its instruction style based on the detected question type (factual, procedural, comparative, ingredient, technique).

**Evaluation** — Responses are scored against gold-standard answers using ROUGE, BLEU, BERTScore, and a token-overlap faithfulness metric that measures grounding in the retrieved context.

---

## Project Structure

```
.
├── 01_corpus_and_benchmark.ipynb   # Corpus loading, validation, and benchmark inspection
├── 02_rag_pipeline.ipynb           # Full pipeline: chunking, embedding, retrieval, generation
├── 03_evaluation.ipynb             # Metric computation and baseline comparison
│
├── background_corpus.json          # 204-document curated knowledge corpus
├── benchmark_dataset.json          # 100-question benchmark with gold answers and metadata
├── input_payload.json              # Query-only input file used by the pipeline
│
├── output_payload.json             # RAG system outputs (responses + retrieved context)
├── test_outputs_baseline.json      # Dense-only baseline outputs for comparison
└── evaluation_results.json         # Per-query and aggregate evaluation scores
```

---

## Setup and Installation

### Prerequisites

- Python 3.9 or later
- A CUDA-capable GPU is recommended for generation (CPU inference is supported via `device_map="auto"` but is significantly slower)

### Install dependencies

```bash
pip install torch
pip install sentence-transformers transformers
pip install rank-bm25 nltk
pip install rouge-score bert-score
pip install gradio
```

Or install all at once:

```bash
pip install torch sentence-transformers transformers rank-bm25 nltk rouge-score bert-score gradio
```

### Download NLTK data

The evaluation notebook requires the NLTK `punkt` tokeniser. This is downloaded automatically when the notebook runs, but you can also install it manually:

```python
import nltk
nltk.download('punkt')
```

### Models

Both models are downloaded automatically from HuggingFace on first run and cached locally:

| Model | Purpose |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Dense retrieval embeddings (384-dim) |
| `Qwen/Qwen2.5-0.5B-Instruct` | Answer generation |

---

## Running the Project

Run the three notebooks in order:

### 1. Corpus and Benchmark

```
01_corpus_and_benchmark.ipynb
```

Loads and validates the 204-document corpus and the 100-question benchmark. Checks that all benchmark answer sources exist in the corpus and previews the input query file.

### 2. RAG Pipeline

```
02_rag_pipeline.ipynb
```

Runs the full pipeline end-to-end:
- Chunks and embeds all corpus documents
- Builds the BM25 index
- Runs hybrid retrieval for each query
- Generates answers using Qwen2.5-0.5B-Instruct
- Saves outputs to `output_payload.json`
- Runs the dense-only baseline and saves to `test_outputs_baseline.json`

An interactive Gradio demo is also launched at the end of this notebook.

### 3. Evaluation

```
03_evaluation.ipynb
```

Scores all outputs against the benchmark using ROUGE, BLEU, BERTScore, and Faithfulness. Computes aggregate metrics and per-query breakdowns. Compares hybrid vs. dense-only baseline. Saves results to `evaluation_results.json`.

---

## Corpus and Benchmark

The corpus contains **204 documents** sourced from Wikipedia and Wikibooks, covering dishes, ingredients, cooking techniques, and regional cuisines across South Asia. All documents include a `doc_id`, `title`, and `full_text`; 21 documents also include structured `sections`.

The benchmark contains **100 manually curated questions**, distributed as follows:

| Question Type | Count |
|---|---|
| Factual       | 48    |
| Procedural    | 18    |
| Comparative   | 17    |
| Ingredient    | 12    |
| Technique     | 5     |

| Difficulty | Count |
|---|---|
| Medium     | 55    |
| Hard       | 24    |
| Easy       | 21    |

Each benchmark item includes the query, a gold-standard answer, the source document ID, question type, and difficulty label.

---

## Retrieval Strategy

The system uses **hybrid retrieval** combining dense and lexical methods:

| Component | Detail |
|---|---|
| Dense model | `all-MiniLM-L6-v2` (384-dim, cosine similarity) |
| Lexical model | `BM25Okapi` with Snowball stemming |
| Candidates per method | 40 |
| Fusion method | Reciprocal Rank Fusion (RRF, k=60) |
| Final top-k returned | 5 |
| Chunking | 220-word max, 40-word overlap, sentence-boundary split |

Documents are prefixed with their title at embedding time (`"passage: {title}. {section}. {text}"`). Queries are prefixed with `"query: "` to align with the asymmetric retrieval format used by MiniLM.

---

## Evaluation Results

Scores are averaged across all 100 benchmark questions. The hybrid system is compared against a dense-only retrieval baseline (same generator, same prompt, dense retrieval only).

| Metric | Hybrid (RAG) | Dense-only (Baseline) | Δ |
|---|---|---|---|
| ROUGE-1 F1   | 0.412 | 0.403 | +0.009 |
| ROUGE-2 F1   | 0.204 | 0.188 | +0.016 |
| ROUGE-L F1   | 0.314 | 0.299 | +0.015 |
| BLEU         | 0.072 | 0.068 | +0.004 |
| BERTScore F1 | 0.892 | 0.889 | +0.003 |
| Faithfulness | 0.908 | 0.891 | +0.017 |

**Notable results:**
- 94 out of 100 responses achieve faithfulness > 0.70, indicating strong grounding in retrieved context
- 33 out of 100 responses achieve perfect faithfulness (1.0)
- The hybrid approach consistently outperforms dense-only retrieval across all six metrics, with the largest gains on ROUGE-2 (+0.016) and Faithfulness (+0.017)

Full per-query scores are stored in `evaluation_results.json`.

---

## Domain Coverage

| Country | Topics covered |
|---|---|
| **India** | Dishes (biryani, butter chicken, dal makhani), ingredients (ghee, paneer, garam masala, tamarind), techniques (dum, tadka, bhuna, tandoor), regional cuisines (Rajasthani, Awadhi, Hyderabadi, Goan, Gujarati, South Indian) |
| **Pakistan** | Karahi, nihari, haleem, chapli kebab, sajji, paya; Punjabi, Sindhi, Pashtun, and Kashmiri traditions; BBQ and grilling culture |
| **Bangladesh** | Hilsa fish, kacchi biryani, panta bhat, bhorta, bakarkhani; mustard oil and panch phoron |
| **Sri Lanka** | Hoppers, kottu roti, lamprais, kiribath, fish ambul thiyal, sambol; Maldive fish, goraka, Ceylon cinnamon, coconut milk |
| **Nepal** | Dal bhat tarkari, momo, gundruk, sel roti, thukpa, dhindo, yomari; Newari cuisine and highland staples |
| **Afghanistan** | Kabuli palaw, mantu, ashak, bolani, aush, qorma; pilaf technique and dried fruit traditions |
| **Bhutan** | Ema datshi, phaksha paa, hoentay; chilli as a vegetable, yak products, red rice, buckwheat |
| **Maldives** | Garudhiya, mas huni, valhomas (Maldive fish); tuna-centric cuisine and breadfruit history |

---

## Limitations and Future Work

### Current limitations

- **Small generator model** — `Qwen2.5-0.5B-Instruct` is lightweight and fast, but its limited capacity leads to truncated or incomplete answers on harder, multi-part questions. A larger model (e.g. 7B+) would likely improve ROUGE and BERTScore substantially.
- **No cross-encoder re-ranking** — The pipeline uses RRF to merge dense and BM25 results but does not apply a cross-encoder re-ranker. Adding one (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a final re-ranking step could improve precision of the top-k chunks passed to the generator.
- **Approximate faithfulness metric** — The faithfulness score uses token overlap between the response and retrieved context. It can overcount function word matches and does not capture paraphrase or inference, meaning it may underreport grounding for fluent, paraphrased answers.
- **Context window capped at 1,800 characters** — For complex or multi-part questions, this limit may exclude relevant retrieved chunks, particularly where the answer spans multiple documents.
- **Corpus coverage gap** — One benchmark answer source (`india_technique_tadka`) is absent from the corpus, meaning at least one question cannot be answered from retrieved context alone.
- **No query expansion** — Queries are used as-is. Techniques such as HyDE (hypothetical document embeddings) or query reformulation could improve recall, particularly for technique and comparative questions.

### Potential improvements

- Replace or supplement BM25 with a learned sparse retriever (e.g. SPLADE) for better domain adaptation
- Expand the corpus to cover more regional sub-cuisines and street food traditions
- Add a query classification step to route questions to specialised retrieval configurations
- Explore few-shot prompting or fine-tuning the generator on South Asian culinary QA pairs
- Implement answer post-verification to flag low-confidence responses rather than returning potentially hallucinated content
