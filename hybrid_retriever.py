import json
import joblib
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "output" / "chunks.json"

TFIDF_MATRIX_FILE = (
    BASE_DIR / "output" / "tfidf_matrix.pkl"
)

TFIDF_VECTORIZER_FILE = (
    BASE_DIR / "output" / "tfidf_vectorizer.pkl"
)

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


print(f"Loaded {len(chunks)} chunks.")


# ---------------------------------------------------------
# Load TF-IDF
# ---------------------------------------------------------

vectorizer = joblib.load(
    TFIDF_VECTORIZER_FILE
)

tfidf_matrix = joblib.load(
    TFIDF_MATRIX_FILE
)


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

embeddings = np.load(
    EMBEDDINGS_FILE
)


print(
    f"TF-IDF matrix shape: {tfidf_matrix.shape}"
)

print(
    f"Embedding matrix shape: {embeddings.shape}"
)


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

print("Model loaded.")


# ---------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------

def retrieve(
    question,
    top_k=5,
    tfidf_weight=0.4,
    embedding_weight=0.6
):

    # -----------------------------------------------------
    # Create TF-IDF vector for question
    # -----------------------------------------------------

    question_tfidf = vectorizer.transform(
        [question]
    )


    # -----------------------------------------------------
    # Calculate TF-IDF similarity
    # -----------------------------------------------------

    tfidf_scores = cosine_similarity(
        question_tfidf,
        tfidf_matrix
    )[0]


    # -----------------------------------------------------
    # Create embedding for question
    # -----------------------------------------------------

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )


    # -----------------------------------------------------
    # Calculate embedding similarity
    # -----------------------------------------------------

    embedding_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]


    # -----------------------------------------------------
    # Combine both scores
    # -----------------------------------------------------

    hybrid_scores = (
        tfidf_weight * tfidf_scores
        +
        embedding_weight * embedding_scores
    )


    # -----------------------------------------------------
    # Get top results
    # -----------------------------------------------------

    top_indices = (
        np.argsort(hybrid_scores)[-top_k:][::-1]
    )


    # -----------------------------------------------------
    # Build results
    # -----------------------------------------------------

    results = []

    for index in top_indices:

        chunk = chunks[index]

        result = {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "policy_id": chunk["policy_id"],
            "section": chunk["section"],
            "pages": chunk["pages"],
            "text": chunk["text"],
            "tfidf_score": float(
                tfidf_scores[index]
            ),
            "embedding_score": float(
                embedding_scores[index]
            ),
            "hybrid_score": float(
                hybrid_scores[index]
            )
        }

        results.append(result)


    return results


# ---------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------

if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a hospital policy question "
            "(or type 'exit'): "
        )


        if question.lower() == "exit":
            break


        results = retrieve(
            question,
            top_k=5
        )


        print("\n")
        print("=" * 75)
        print("TOP HYBRID RETRIEVAL RESULTS")
        print("=" * 75)


        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {rank}"
            )

            print("-" * 75)

            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.4f}"
            )

            print(
                f"TF-IDF Score: "
                f"{result['tfidf_score']:.4f}"
            )

            print(
                f"Embedding Score: "
                f"{result['embedding_score']:.4f}"
            )

            print(
                f"Document: "
                f"{result['document']}"
            )

            print(
                f"Policy ID: "
                f"{result['policy_id']}"
            )

            print(
                f"Section: "
                f"{result['section']}"
            )

            print(
                f"Pages: "
                f"{result['pages']}"
            )

            print(
                f"\n{result['text']}"
            )