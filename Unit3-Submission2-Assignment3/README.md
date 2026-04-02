# Unit 3 Assignment: Building a Production Advanced RAG System

**Topic:** Advanced RAG - Retrieval Enhancement, Re-Ranking, and Query Expansion  
**Tools:** Python, HuggingFace, Groq API, rank-bm25, sentence-transformers, Dotenv

## Overview

This folder contains a `.ipynb` file which builds a **Production-Grade Advanced RAG (Retrieval Augmented Generation) System** for a university knowledge assistant. It goes significantly beyond a Naïve RAG pipeline by combining multiple retrieval and query expansion techniques into a single end-to-end system. The system takes a student's vague, short question and retrieves the most relevant AI/ML documents to generate a precise, grounded answer. It also contains a pdf file which contains the screenshots of all the outputs along with the required comparison table.

## Parts Covered

- **Part 1 - Document Corpus:** A custom corpus of 15 AI/ML documents, designed to expose vocabulary gaps between student queries and technical documents. Includes proper nouns (BM25, AdamW, GPTQ) and 3 documents covering neural network training from distinct angles.
- **Part 2 - Hybrid Retrieval:** A `HybridRetriever` class combining BM25 (sparse) and SBERT (dense) retrieval using Reciprocal Rank Fusion (RRF). Returns `bm25_rank`, `sbert_rank`, and `rrf_score` for every result.
- **Part 3 - Cross-Encoder Re-Ranking:** A `rerank()` function using `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score retrieved candidates with higher precision using the original user query.
- **Part 4 - Query Expansion:** Both HyDE (Hypothetical Document Embedding) and Multi-Query expansion implemented using Groq (Llama-3.1-8b-instant) as the LLM.
- **Part 5 - End-to-End Pipeline:** A single `advanced_rag()` function wiring Query Expansion → Hybrid Retrieval → Re-Ranking → LLM Generation together.
- **Part 6 - Comparison Experiment:** A before/after comparison table showing Naïve RAG vs Advanced RAG across 3 test queries with genuine observations.

## Bonus Challenges

- **Bonus 1 - Weighted RRF:** Modified RRF formula with α ∈ {0.3, 0.5, 0.7} to control BM25 vs SBERT contribution. Experiments show higher α improves keyword-heavy queries while lower α improves semantic queries.
- **Bonus 2 - Chunk Size Study:** A longer document split into 50, 100, and 200-word chunks. Cross-encoder scores used to measure how chunk size affects retrieval precision.
- **Bonus 3 - ColBERT as Third Retriever:** ColBERT MaxSim scoring added as a third retriever in a `TripleHybridRetriever`, fusing BM25 + SBERT + ColBERT with three-way RRF.
