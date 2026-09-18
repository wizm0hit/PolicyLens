import json
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "output" / "chunks.json"

OUTPUT_FOLDER = BASE_DIR / "output"

TFIDF_MATRIX_FILE = OUTPUT_FOLDER / "tfidf_matrix.pkl"

TFIDF_VECTORIZER_FILE = (
    OUTPUT_FOLDER / "tfidf_vectorizer.pkl"
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


print(f"Loaded {len(chunks)} chunks.")


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
# Create TF-IDF vectorizer
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)


# ---------------------------------------------------------
# Convert chunks into TF-IDF vectors
# ---------------------------------------------------------

tfidf_matrix = vectorizer.fit_transform(texts)


print(
    f"TF-IDF matrix shape: {tfidf_matrix.shape}"
)


# ---------------------------------------------------------
# Save vectorizer
# ---------------------------------------------------------

joblib.dump(
    vectorizer,
    TFIDF_VECTORIZER_FILE
)


# ---------------------------------------------------------
# Save TF-IDF matrix
# ---------------------------------------------------------

joblib.dump(
    tfidf_matrix,
    TFIDF_MATRIX_FILE
)


print("\nTF-IDF processing complete!")

print(
    f"Vectorizer saved to: "
    f"{TFIDF_VECTORIZER_FILE}"
)

print(
    f"Matrix saved to: "
    f"{TFIDF_MATRIX_FILE}"
)