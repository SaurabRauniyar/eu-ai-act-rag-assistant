# Embedding Model Comparison

Two embedding models were evaluated using the same:

- EU AI Act dataset
- 160 document chunks
- 12 evaluation questions
- FAISS vector search
- Cosine similarity
- Top-K = 5

## Retrieval Results

| Test | Expected Article | MiniLM Rank | BGE Rank |
|---|---|---:|---:|
| Provider obligations | Article 16 | 4 | 2 |
| Human oversight | Article 14 | 1 | 1 |
| Deployer obligations | Article 26 | 1 | 1 |
| Prohibited AI practices | Article 5 | ❌ Not Top 5 | 1 |
| Risk management | Article 9 | 1 | 3 |
| Data governance | Article 10 | 1 | 1 |
| Technical documentation | Article 11 | ❌ Not Top 5 | 2 |
| Record-keeping and logging | Article 12 | 1 | 1 |
| Accuracy, robustness and cybersecurity | Article 15 | 1 | 1 |
| Transparency | Article 50 | 1 | 1 |
| General-purpose AI obligations | Article 53 | 1 | 1 |
| Post-market monitoring | Article 72 | 1 | 1 |

## Overall Comparison

| Metric | MiniLM | BGE |
|---|---:|---:|
| Embedding dimension | 384 | 384 |
| Hit@1 | **75% (9/12)** | **75% (9/12)** |
| Hit@3 | **75% (9/12)** | **100% (12/12)** |
| Hit@5 | **83.3% (10/12)** | **100% (12/12)** |
| Top-5 failures | **2** | **0** |

## Model Selection

Both models achieved the same Hit@1 score of 75%.

However, BGE retrieved the expected article within the Top 3 for all
12 questions and had no Top-5 retrieval failures.

MiniLM failed to retrieve the expected article within the Top 5 for
two questions:

- Article 5 — Prohibited AI practices
- Article 11 — Technical documentation

Based on this evaluation set, `BAAI/bge-small-en-v1.5` was selected
as the embedding model for the next stage of the RAG pipeline.

These results are based on a small manually constructed evaluation
set and should not be interpreted as general benchmark performance
for either embedding model.