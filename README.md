# south-asian-rag-assistant

A Retrieval-Augmented Generation (RAG) based culinary assistant specialised in South Asian cuisine, designed to answer food-related queries using a custom-built knowledge corpus and evaluation pipeline.

---

## Full Description

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** question-answering system focused on South Asian cuisine. The system retrieves relevant information from a curated corpus and generates context-aware answers using a language model.

Developed as part of **COMP64702: Transforming Text Into Meaning**, the project focuses on designing and evaluating an effective RAG pipeline.

---

## Key Features

- Custom background corpus built from curated public sources (e.g., Wikipedia, Wikibooks)  
- Hybrid retrieval combining:
  - Dense retrieval (SentenceTransformers embeddings)
  - Lexical retrieval (BM25)  
- Chunking and embedding pipeline for efficient document representation  
- LLM-based answer generation using Qwen2.5-0.5B-Instruct  
- Evaluation pipeline using standard NLP metrics  

---

## System Overview

The system follows a standard RAG architecture:

- **Ingestion** – Preprocessing, chunking, and embedding of corpus data  
- **Retrieval** – Hybrid retrieval (semantic + keyword-based) to find relevant context  
- **Generation** – Producing answers using an LLM with retrieved context  
- **Evaluation** – Measuring performance against benchmark datasets  

---

## Retrieval Strategy

The system uses **hybrid retrieval**, combining:

- **Dense retrieval** → captures semantic similarity  
- **BM25 (lexical retrieval)** → captures exact keyword matches  

This improves robustness and recall compared to using dense retrieval alone.

---

## Evaluation

The system is evaluated on a benchmark dataset using:

- ROUGE-1  
- ROUGE-2  
- ROUGE-L  
- BLEU  
- BERTScore  
- Faithfulness  

Results are aggregated and stored in `evaluation_results.json`.

---

## Project Structure
01_corpus_and_benchmark.ipynb
02_rag_pipeline.ipynb
03_evaluation.ipynb

background_corpus.json
benchmark_dataset.json
benchmark_input_only.json

test_outputs.json
evaluation_results.json

---

## Domain Focus

The assistant is specialized in South Asian cuisine, covering:

- Indian, Pakistani, Bangladeshi, Sri Lankan, and Nepali dishes  
- Ingredients, spices, and cooking techniques  
- Regional variations and cultural context  

---

## Running the Project

1. Install dependencies:
pip install -r requirements.txt

2. Run notebooks in order:
01_corpus_and_benchmark.ipynb
02_rag_pipeline.ipynb
03_evaluation.ipynb

---

## Notes

- Models are downloaded from HuggingFace on first run and cached locally  
- Designed to run in both local and notebook environments  