"""
Create BGE embeddings for the EU AI Act chunks.

Model:
    BAAI/bge-small-en-v1.5

Input:
    data/processed/eu_ai_act_chunks.json

Output:
    data/processed/eu_ai_act_bge_embeddings.npy
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

CHUNKS_PATH = Path("data/processed/eu_ai_act_chunks.json")
EMBEDDINGS_PATH = Path(
    "data/processed/eu_ai_act_bge_embeddings.npy"
)


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks)}")


# ---------------------------------------------------------
# Load BGE embedding model
# ---------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("BGE embedding model loaded.")


# ---------------------------------------------------------
# Prepare chunk texts
# ---------------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

print(f"Texts prepared for embedding: {len(texts)}")


# ---------------------------------------------------------
# Create embeddings
# ---------------------------------------------------------

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print("BGE embeddings created.")
print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

np.save(EMBEDDINGS_PATH, embeddings)

print(f"Embeddings saved to: {EMBEDDINGS_PATH}")