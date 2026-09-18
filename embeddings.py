import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "output" / "chunks.json"

EMBEDDINGS_FILE = (
    BASE_DIR / "output" / "embeddings.npy"
)


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    f"Loaded {len(chunks)} chunks."
)


# ---------------------------------------------------------
# Prepare text
# ---------------------------------------------------------

texts = []

for chunk in chunks:

    text = (
        f"Policy: {chunk['document']}\n"
        f"Section: {chunk['section']}\n"
        f"{chunk['text']}"
    )

    texts.append(text)


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

try:
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        local_files_only=True
    )
except Exception:
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

print("Embedding model loaded.")


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# Convert to NumPy array
# ---------------------------------------------------------

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ---------------------------------------------------------
# Display shape
# ---------------------------------------------------------

print(
    f"\nEmbedding shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

np.save(
    EMBEDDINGS_FILE,
    embeddings
)


print(
    f"\nEmbeddings saved to:"
)

print(
    EMBEDDINGS_FILE
)