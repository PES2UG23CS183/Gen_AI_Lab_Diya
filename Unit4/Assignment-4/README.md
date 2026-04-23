
# Unit 4 Submission 2 - Assignment 4: Evaluated Agentic RAG System

**Tools:** Python, CrewAI, DeepEval, FAISS, HuggingFace, Groq API, LangChain, sentence-transformers

## Overview

This folder contains a `.ipynb` file which builds a self-evaluating agentic RAG system using CrewAI. It goes beyond a standard RAG pipeline by adding an autonomous quality evaluation and revision loop - if the generated answer fails quality thresholds, a third agent automatically rewrites it. The system takes a user question, retrieves relevant context from a FAISS vector store, generates an answer, evaluates it with DeepEval metrics, and revises it if needed. It also contains a `pdf` file with screenshots of all outputs and the results table.


## Parts Covered

* **Part 1 - Knowledge Base:** A 600+ word space exploration corpus covering the Solar System, Mars exploration, JWST, black holes, and space history. Split into 17 overlapping chunks using `RecursiveCharacterTextSplitter` and indexed with FAISS + `all-MiniLM-L6-v2` embeddings.

* **Part 2 - RAG Agent:** A CrewAI agent with a `@tool`-decorated FAISS retriever that outputs both the answer and the retrieved context in a structured `ANSWER: / CONTEXT:` format, ensuring the evaluator has everything it needs.

* **Part 3 - Quality Evaluator Agent:** A CrewAI agent wrapping DeepEval's `FaithfulnessMetric` and `AnswerRelevancyMetric` with a threshold of 0.7. Uses a custom `GroqJudge` class as the judge LLM and returns JSON scores, a PASS/FAIL verdict, and specific failure reasons.

* **Part 4 - Revisor Agent:** Activates only on FAIL. Receives the original question, failed answer, evaluator's specific failure reasons, and retrieved context, then rewrites the answer to address each identified issue while staying strictly grounded in the source.

* **Part 5 - Full Pipeline:** A `run_full_pipeline()` function wiring all three agents together sequentially. Tested on 5 knowledge-base questions and 2 adversarial questions (answers not in the KB), with a final results table showing initial vs. final scores.


## Results Summary

| Metric | Value |
|---|---|
| Initial pass rate | 6/7 (86%) |
| Final pass rate | 6/7 (86%) |
| Questions revised | 1 |
| Avg faithfulness (KB questions) | 1.00 |
| Avg relevancy (KB questions) | 0.93 |


## Note on Rate Limit Error

> The notebook contains a visible `RateLimitError` in the cell running question 7 ("How does nuclear fusion work inside a star?"). This occurred because the Groq free-tier daily token limit (100,000 TPD) was exhausted after running 6 questions back-to-back. The error is not a code issue - the pipeline code is identical to the cells that successfully ran questions 1–6. Question 7 was successfully completed in a subsequent cell after switching to a different model within the same API key, and its result is included in the final results table.


## Key Observations

* All 4 straightforward knowledge-base questions passed with perfect scores (F=1.00, R=1.00), confirming the RAG retrieval and generation pipeline works correctly for factual queries.
* Question 5 (ISS inhabitation duration) failed on relevancy (R=0.33) despite perfect faithfulness — the agent retrieved correct facts but didn't directly answer what was asked. The revisor improved scores but could not fully fix the gap.
* Both adversarial questions were handled gracefully — the system either stated the information was not in the knowledge base (faithful, honest response) or retrieved loosely related context and still scored well.
* The revisor demo showed clear improvement: a deliberately vague answer (F=0.0, R=0.0) was revised to a grounded, specific answer scoring F=1.00, R=1.00, confirming the revision mechanism works as intended.
