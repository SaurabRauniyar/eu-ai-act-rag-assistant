"""
Evaluate MiniLM retrieval on the EU AI Act.

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Retrieval:
    FAISS cosine similarity

Evaluation:
    Uses questions from evaluation/retrieval_questions.json
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

CHUNKS_PATH = Path(
    "data/processed/eu_ai_act_chunks.json"
)

INDEX_PATH = Path(
    "data/processed/eu_ai_act_faiss.index"
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
# Load evaluation questions
# ---------------------------------------------------------

with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    test_questions = json.load(file)

print(f"Evaluation questions loaded: {len(test_questions)}")


# ---------------------------------------------------------
# Load MiniLM FAISS index
# ---------------------------------------------------------

index = faiss.read_index(str(INDEX_PATH))

print(f"Vectors in FAISS: {index.ntotal}")


# ---------------------------------------------------------
# Load MiniLM embedding model
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("MiniLM embedding model loaded.")


# ---------------------------------------------------------
# Run retrieval evaluation
# ---------------------------------------------------------

TOP_K = 5

for test_number, test in enumerate(test_questions, start=1):

    question = test["question"]
    expected = test["expected"]

    # -----------------------------------------------------
    # Convert question into a vector
    # -----------------------------------------------------

    query_embedding = model.encode([question])

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # -----------------------------------------------------
    # Normalize query vector
    #
    # The document vectors were also normalized when
    # creating the FAISS index.
    #
    # Normalized vectors + inner product gives us
    # cosine similarity.
    # -----------------------------------------------------

    faiss.normalize_L2(query_embedding)


    # -----------------------------------------------------
    # Search for the 5 most similar chunks
    # -----------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )


    # -----------------------------------------------------
    # Display question
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print(f"TEST {test_number}")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print(f"\nExpected: {expected}")

    print("\nTop retrieved sections:")


    # -----------------------------------------------------
    # Check where the expected Article appears
    # -----------------------------------------------------

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

        # Save the rank if this is the expected Article
        if section == expected and found_rank is None:
            found_rank = rank


    # -----------------------------------------------------
    # PASS / FAIL
    # -----------------------------------------------------

    if found_rank is not None:

        print(
            f"\nPASS - {expected} "
            f"retrieved at Rank {found_rank}"
        )

    else:

        print(
            f"\nFAIL - {expected} "
            f"not found in Top {TOP_K}"
        )