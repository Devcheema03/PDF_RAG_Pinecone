from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


# Load the embedding model
model = SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(texts):
    """
    Convert text into numerical vectors.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()