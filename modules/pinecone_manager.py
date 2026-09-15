import os
from pinecone import Pinecone, ServerlessSpec


# -----------------------------
# Pinecone Settings
# -----------------------------

INDEX_NAME = "pdf-rag-index"
DIMENSION = 384


# -----------------------------
# Connect to Pinecone
# -----------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)


# -----------------------------
# Create / Connect to Index
# -----------------------------

def create_index():
    """
    Create the Pinecone index if it does not already exist.
    If it already exists, connect to it.
    """

    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(INDEX_NAME)


# -----------------------------
# Store Vectors in Pinecone
# -----------------------------

def upsert_vectors(index, embeddings, chunks, document_name):
    """
    Store embeddings and their metadata in Pinecone.
    """

    vectors = []

    for i, (embedding, chunk) in enumerate(
        zip(embeddings, chunks)
    ):

        vector = {
            "id": f"{document_name}_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "document_name": document_name
            }
        }

        vectors.append(vector)

    index.upsert(vectors=vectors)
