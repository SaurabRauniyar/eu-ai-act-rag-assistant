"""
Create a FAISS index for BGE embeddings.

Input:
    data/processed/eu_ai_act_bge_embeddings.npy

Output:
    data/processed/eu_ai_act_bge_faiss.index
"""

from pathlib import Path

import faiss
import numpy as np


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

EMBEDDINGS_PATH = Path(
    "data/processed/eu_ai_act_bge_embeddings.npy"
)

INDEX_PATH = Path(
    "data/processed/eu_ai_act_bge_faiss.index"
)


# ---------------------------------------------------------
# Load BGE embeddings
# ---------------------------------------------------------

embeddings = np.load(EMBEDDINGS_PATH)

print("BGE embeddings loaded.")
print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------------
# Prepare embeddings for cosine similarity
# ---------------------------------------------------------

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)


# ---------------------------------------------------------
# Create FAISS index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

print("FAISS index created.")
print("Vector dimension:", dimension)


# ---------------------------------------------------------
# Add vectors
# ---------------------------------------------------------

index.add(embeddings)

print("Vectors added to FAISS:", index.ntotal)


# ---------------------------------------------------------
# Save index
# ---------------------------------------------------------

faiss.write_index(index, str(INDEX_PATH))

print(f"BGE FAISS index saved to: {INDEX_PATH}")