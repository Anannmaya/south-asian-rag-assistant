# south-asian-rag-assistant
A Retrieval-Augmented Generation (RAG) based culinary assistant specialised in South Asian cuisine, designed to answer food-related queries using a custom-built knowledge corpus and evaluation pipeline

## Full Description

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) based question-answering system focused on South Asian cuisine. The system acts as an AI-powered culinary assistant capable of answering user queries related to dishes, ingredients, cooking methods, and regional food knowledge.

The project is developed as part of the COMP64702: Transforming Text Into Meaning coursework, where the goal is to design and evaluate a bespoke RAG framework.

---

## Key Features

- Custom Background Corpus built from curated public sources (e.g., Wikipedia, Wikibooks)  
- Chunking and Embedding Pipeline for efficient text representation  
- Semantic Retrieval and Ranking using vector similarity  
- LLM-based Answer Generation using Qwen2.5-0.5B-Instruct  
- Evaluation Pipeline to assess system performance against benchmark datasets  

---

## System Overview

The system follows a standard RAG architecture:

- **Ingestion** – Preprocessing and embedding of culinary knowledge  
- **Retrieval** – Finding relevant context for a given query  
- **Generation** – Producing answers using an LLM with retrieved context  
- **Evaluation** – Measuring accuracy and relevance of responses  

---

## Domain Focus

The assistant is specialized in South Asian cuisine, covering:

- Indian, Pakistani, Bangladeshi, Sri Lankan, and Nepali dishes  
- Ingredients, spices, and cooking techniques  
- Regional variations and cultural context  

---

