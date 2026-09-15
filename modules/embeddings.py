from sentence_transformers import SentenceTransformer


# -----------------------------
# Embedding Model
# -----------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Load the embedding model
model = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# Generate Embeddings
# -----------------------------

def generate_embeddings(texts):
    """
    Convert text into numerical vectors.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()
