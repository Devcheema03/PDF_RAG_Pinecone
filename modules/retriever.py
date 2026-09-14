from modules.embeddings import generate_embeddings


def retrieve_chunks(index, query, top_k=5, similarity_threshold=0.30):
    """
    Search Pinecone for chunks relevant to the user's question.
    """

    # Convert the user's question into a vector
    query_embedding = generate_embeddings([query])[0]

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    relevant_chunks = []

    for match in results["matches"]:

        score = match["score"]

        # Only keep sufficiently similar results
        if score >= similarity_threshold:

            relevant_chunks.append({
                "text": match["metadata"]["text"],
                "page_number": match["metadata"]["page_number"],
                "chunk_id": match["metadata"]["chunk_id"],
                "document_name": match["metadata"]["document_name"],
                "score": score
            })

    return relevant_chunks