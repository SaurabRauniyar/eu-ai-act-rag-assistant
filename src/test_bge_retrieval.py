"""
Test BGE semantic retrieval on the EU AI Act.

Embedding model:
    BAAI/bge-small-en-v1.5

Retrieval:
    FAISS cosine similarity
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

CHUNKS_PATH = Path("data/processed/eu_ai_act_chunks.json")

INDEX_PATH = Path(
    "data/processed/eu_ai_act_bge_faiss.index"
)

QUESTIONS_PATH = Path(
    "evaluation/retrieval_questions.json"
)


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks)}")


# ---------------------------------------------------------
# Load BGE FAISS index
# ---------------------------------------------------------

index = faiss.read_index(str(INDEX_PATH))

print(f"Vectors in FAISS: {index.ntotal}")


# ---------------------------------------------------------
# Load SAME BGE embedding model
# ---------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

print("BGE embedding model loaded.")


# ---------------------------------------------------------
# Evaluation questions
# ---------------------------------------------------------

# ---------------------------------------------------------
# Load evaluation questions
# ---------------------------------------------------------

with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    test_questions = json.load(file)

print(f"Evaluation questions loaded: {len(test_questions)}")


# ---------------------------------------------------------
# Run retrieval tests
# ---------------------------------------------------------

TOP_K = 5

for test_number, test in enumerate(test_questions, start=1):

    question = test["question"]
    expected = test["expected"]

    # Convert question to embedding
    query_embedding = model.encode([question])

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    print("\n" + "=" * 70)
    print(f"TEST {test_number}")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print(f"\nExpected: {expected}")

    print("\nTop retrieved sections:")

    found_rank = None

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        chunk = chunks[idx]

        section = chunk["section"]

        print(
            f"Rank {rank}: "
            f"{section} "
            f"(score: {score:.4f})"
        )

        if section == expected and found_rank is None:
            found_rank = rank

    # Evaluation result
    if found_rank is not None:
        print(
            f"\nPASS - {expected} retrieved at Rank {found_rank}"
        )
    else:
        print(
            f"\nFAIL - {expected} not found in Top {TOP_K}"
        )