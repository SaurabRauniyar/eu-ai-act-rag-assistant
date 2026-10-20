"""
Create a FAISS index for the EU AI Act embeddings.

Input:
    data/processed/eu_ai_act_embeddings.npy

Output:
    data/processed/eu_ai_act_faiss.index
"""

from pathlib import Path

import faiss
import numpy as np


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

EMBEDDINGS_PATH = Path("data/processed/eu_ai_act_embeddings.npy")
INDEX_PATH = Path("data/processed/eu_ai_act_faiss.index")


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

embeddings = np.load(EMBEDDINGS_PATH)

print("Embeddings loaded.")
print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------------
# Prepare embeddings for FAISS
# ---------------------------------------------------------

embeddings = embeddings.astype("float32")

# Normalize vectors so inner product can be used
# as cosine similarity.
faiss.normalize_L2(embeddings)


# ---------------------------------------------------------
# Create FAISS index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

print("FAISS index created.")
print("Vector dimension:", dimension)


# ---------------------------------------------------------
# Add embeddings to FAISS
# ---------------------------------------------------------

index.add(embeddings)

print("Vectors added to FAISS:", index.ntotal)


# ---------------------------------------------------------
# Save FAISS index
# ---------------------------------------------------------

faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index saved to: {INDEX_PATH}")