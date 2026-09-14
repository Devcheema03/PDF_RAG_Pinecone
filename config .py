import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# -----------------------------
# Pinecone
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


OLLAMA_MODEL = "llama3.2:3b"


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# RAG Settings
# -----------------------------
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.30


MAX_FILE_SIZE_MB = 20