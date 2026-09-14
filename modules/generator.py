import requests

from config import OLLAMA_MODEL


OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_answer(query, retrieved_chunks):
    """
    Generate an answer using only the retrieved PDF chunks.
    """

    if not retrieved_chunks:
        return {
            "answer": "The answer is not available in the provided document.",
            "sources": []
        }

    # Build context from retrieved PDF chunks
    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"\n--- Page {chunk['page_number']} ---\n"
            f"{chunk['text']}\n"
        )

    # Create prompt
    prompt = f"""
You are a PDF question answering assistant.

IMPORTANT RULES:
1. Answer ONLY from the DOCUMENT CONTEXT below.
2. Do NOT use outside knowledge.
3. If the answer is present in the context, answer it.
4. If the answer is not present, say exactly:
The answer is not available in the provided document.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            answer = "The answer is not available in the provided document."

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }

    except Exception as error:

        return {
            "answer": f"Error generating answer: {error}",
            "sources": retrieved_chunks
        }