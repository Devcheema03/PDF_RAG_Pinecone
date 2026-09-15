import os
from groq import Groq


# -----------------------------
# Groq Settings
# -----------------------------

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


MODEL_NAME = "llama-3.1-8b-instant"


# -----------------------------
# Generate Answer
# -----------------------------

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
3. Do NOT make up information.
4. If the answer is present in the context, answer clearly.
5. If the answer is not present in the context, say exactly:
The answer is not available in the provided document.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions strictly "
                        "from the provided document context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

        if not answer:
            answer = (
                "The answer is not available "
                "in the provided document."
            )

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }

    except Exception as error:

        return {
            "answer": f"Error generating answer: {error}",
            "sources": retrieved_chunks
        }
