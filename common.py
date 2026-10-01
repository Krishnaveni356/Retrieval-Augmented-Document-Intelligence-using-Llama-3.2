"""Shared config + helpers. Every knob can be overridden in .env or the environment."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
DB_DIR = os.getenv("DB_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters shared between neighbours
TOP_K = int(os.getenv("TOP_K", "4"))                    # chunks sent to the LLM
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.2"))        # cosine similarity floor
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "anthropic" | "ollama"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_embedder = None
def embed(texts):
    """Text -> normalized vectors (runs locally, no API key needed)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
def get_collection():
    """Persistent local vector store (cosine distance)."""
    import chromadb
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
