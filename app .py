import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from modules.pdf_loader import extract_text_from_pdf
from modules.chunker import chunk_pages
from modules.embeddings import generate_embeddings
from modules.pinecone_manager import create_index, upsert_vectors
from modules.retriever import retrieve_chunks
from modules.generator import generate_answer


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #172554);
    color: white;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 5px;
}

.developer {
    text-align: center;
    color: #60a5fa;
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 30px;
}

.card {
    background: rgba(30, 41, 59, 0.85);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    margin-bottom: 20px;
}

.answer-card {
    background: rgba(15, 23, 42, 0.95);
    padding: 25px;
    border-radius: 18px;
    border-left: 5px solid #60a5fa;
    margin-top: 15px;
    font-size: 17px;
    line-height: 1.7;
}

.answer-title {
    color: #60a5fa;
    font-size: 21px;
    font-weight: bold;
    margin-bottom: 10px;
}

.source-card {
    background: rgba(30, 41, 59, 0.8);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    margin-bottom: 15px;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 30px;
    font-size: 13px;
}

.footer-name {
    color: #60a5fa;
    font-weight: bold;
}

section[data-testid="stSidebar"] {
    background: #0b1120;
}

.stButton > button {
    border-radius: 12px;
    font-weight: bold;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">📚 PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Document Question Answering using RAG'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="developer">'
    'Developed by Abubakkar Cheema'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Session State
# -----------------------------

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "query_history" not in st.session_state:
    st.session_state["query_history"] = []


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("⚙️ RAG Settings")

    st.write(
        "Customize how your RAG system processes "
        "and retrieves information."
    )

    st.divider()

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1000,
        value=500,
        step=50
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10
    )

    st.divider()

    st.subheader("🤖 AI System")

    st.write("🧠 LLM: **Llama 3.2 3B**")
    st.write("🔢 Embeddings: **MiniLM-L6-v2**")
    st.write("🗄️ Vector DB: **Pinecone**")
    st.write("📄 PDF Engine: **PyMuPDF**")
    st.write("👁️ OCR: **Tesseract**")
    st.write("🌐 UI: **Streamlit**")


# -----------------------------
# Upload Section
# -----------------------------

st.markdown(
    '<div class="section-title">📤 Upload Your Documents</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">
        <b>Upload PDF documents to begin</b><br>
        <span style="color:#94a3b8;">
        Maximum 20 MB per file • Multiple PDFs supported
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Choose PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True
)


# -----------------------------
# File Size Check
# -----------------------------

if uploaded_files:

    valid_files = []

    for uploaded_file in uploaded_files:

        file_size_mb = uploaded_file.size / (
            1024 * 1024
        )

        if file_size_mb <= 20:

            valid_files.append(uploaded_file)

        else:

            st.error(
                f"{uploaded_file.name} is too large. "
                f"Maximum size is 20 MB."
            )

    uploaded_files = valid_files


# -----------------------------
# Process PDFs
# -----------------------------

if uploaded_files:

    st.markdown(
        '<div class="section-title">📄 Document Processing</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 Process PDF(s)",
        use_container_width=True
    ):

        with st.spinner(
            "Processing PDF documents..."
        ):

            all_chunks = []

            for uploaded_file in uploaded_files:

                file_path = (
                    f"data/{uploaded_file.name}"
                )

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                # Extract PDF text / OCR
                pages = extract_text_from_pdf(
                    file_path
                )

                # Create chunks
                chunks = chunk_pages(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # Add document name
                for chunk in chunks:

                    chunk["document_name"] = (
                        uploaded_file.name
                    )

                all_chunks.extend(chunks)

            st.session_state["chunks"] = (
                all_chunks
            )

        st.success(
            f"✅ PDF processing complete! "
            f"{len(all_chunks)} chunks created."
        )

        # -----------------------------
        # Generate Embeddings
        # -----------------------------

        with st.spinner(
            "Generating embeddings..."
        ):

            texts = [
                chunk["text"]
                for chunk in all_chunks
            ]

            embeddings = generate_embeddings(
                texts
            )

        # -----------------------------
        # Pinecone
        # -----------------------------

        with st.spinner(
            "Storing vectors in Pinecone..."
        ):

            index = create_index()

            upsert_vectors(
                index,
                embeddings,
                all_chunks,
                "uploaded_documents"
            )

        st.success(
            "✅ Embeddings created and stored in Pinecone!"
        )


# -----------------------------
# System Status
# -----------------------------

if st.session_state["chunks"]:

    st.markdown(
        '<div class="section-title">📊 System Status</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        document_count = len(
            set(
                chunk["document_name"]
                for chunk in st.session_state["chunks"]
            )
        )

        st.metric(
            "Documents",
            document_count
        )

    with col2:

        st.metric(
            "Text Chunks",
            len(st.session_state["chunks"])
        )

    with col3:

        st.metric(
            "System",
            "Ready"
        )


# -----------------------------
# Ask Question
# -----------------------------

st.divider()

st.markdown(
    '<div class="section-title">💬 Ask Your Document</div>',
    unsafe_allow_html=True
)

query = st.text_input(
    "Enter your question",
    placeholder="Example: What is OpenFlow?"
)


if query:

    if not st.session_state["chunks"]:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

    else:

        if st.button(
            "🔍 Search Document",
            use_container_width=True
        ):

            with st.spinner(
                "Searching the document..."
            ):

                index = create_index()

                retrieved_chunks = retrieve_chunks(
                    index=index,
                    query=query,
                    top_k=top_k,
                    similarity_threshold=(
                        similarity_threshold
                    )
                )

            # -----------------------------
            # No Relevant Information
            # -----------------------------

            if not retrieved_chunks:

                answer = (
                    "The answer is not available "
                    "in the provided document."
                )

                st.warning(answer)

                st.session_state[
                    "query_history"
                ].append({
                    "question": query,
                    "answer": answer
                })

            else:

                st.success(
                    f"✅ Found "
                    f"{len(retrieved_chunks)} "
                    f"relevant chunk(s)."
                )

                # -----------------------------
                # Generate Answer
                # -----------------------------

                with st.spinner(
                    "Generating answer with Llama 3.2..."
                ):

                    result = generate_answer(
                        query=query,
                        retrieved_chunks=(
                            retrieved_chunks
                        )
                    )

                answer = result["answer"]

                # -----------------------------
                # Answer
                # -----------------------------

                st.markdown(
                    f"""
                    <div class="answer-card">

                        <div class="answer-title">
                            🤖 AI Answer
                        </div>

                        {answer}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # -----------------------------
                # Retrieved Sources
                # -----------------------------

                st.markdown(
                    '<div class="section-title">'
                    '📚 Retrieved Sources'
                    '</div>',
                    unsafe_allow_html=True
                )

                for number, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <b>📌 Source {number}</b>

                        <br><br>

                        📄 <b>Document:</b>
                        {chunk["document_name"]}

                        <br>

                        📖 <b>Page:</b>
                        {chunk["page_number"]}

                        <br>

                        📊 <b>Similarity Score:</b>
                        {chunk["score"]:.3f}

                        <br><br>

                        📝 <b>Excerpt:</b>

                        <br>

                        {chunk["text"][:500]}...

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Save history
                st.session_state[
                    "query_history"
                ].append({
                    "question": query,
                    "answer": answer
                })


# -----------------------------
# Query History
# -----------------------------

if st.session_state["query_history"]:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🕘 Query History'
        '</div>',
        unsafe_allow_html=True
    )

    for number, item in enumerate(
        reversed(
            st.session_state["query_history"]
        ),
        start=1
    ):

        with st.expander(
            f"Question {number}: {item['question']}"
        ):

            st.markdown(
                f"""
                <div class="answer-card">

                    <div class="answer-title">
                        🤖 Answer
                    </div>

                    {item["answer"]}

                </div>
                """,
                unsafe_allow_html=True
            )


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    """
    <div class="footer">

        📚 PDF RAG Assistant

        <br><br>

        Powered by Python • Streamlit • Pinecone •
        Sentence Transformers • Ollama

        <br><br>

        Developed by
        <span class="footer-name">
        Abubakkar Cheema
        </span>

    </div>
    """,
    unsafe_allow_html=True
)
