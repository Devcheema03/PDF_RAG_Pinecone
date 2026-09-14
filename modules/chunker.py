from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages, chunk_size=500, chunk_overlap=50):
    """
    Split PDF text into smaller chunks.

    chunk_size:
        Maximum size of each chunk.

    chunk_overlap:
        Number of characters shared between neighboring chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk_number, text in enumerate(
            page_chunks,
            start=1
        ):

            chunks.append({
                "text": text,
                "page_number": page["page_number"],
                "chunk_id": (
                    f"page_{page['page_number']}"
                    f"_chunk_{chunk_number}"
                )
            })

    return chunks