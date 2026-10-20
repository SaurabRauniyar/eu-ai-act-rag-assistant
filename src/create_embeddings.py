"""
Create embeddings for the EU AI Act chunks.

Input:
    data/processed/eu_ai_act_chunks.json

Output:
    data/processed/eu_ai_act_embeddings.npy
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

CHUNKS_PATH = Path("data/processed/eu_ai_act_chunks.json")
EMBEDDINGS_PATH = Path("data/processed/eu_ai_act_embeddings.npy")


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks)}")


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully.")


# ---------------------------------------------------------
# Extract text from chunks
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

print("Embeddings created successfully.")
print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

np.save(EMBEDDINGS_PATH, embeddings)

print(f"Embeddings saved to: {EMBEDDINGS_PATH}")