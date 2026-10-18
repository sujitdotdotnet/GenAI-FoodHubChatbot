"""
Pre-build the policy vector index.

Run at Docker build time so the embedding model is already downloaded and the ChromaDB index
already exists when the Space starts (fast first start). Also usable locally:

    python build_index.py
"""
import sys
import time
from pathlib import Path

from policy_rag import PolicyRetriever

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    started = time.time()
    retriever = PolicyRetriever.from_directory(BASE_DIR / "policies", persist_dir=BASE_DIR / "chroma_db")
    info = retriever.describe()
    print(f"Documents : {info['documents']}")
    print(f"Chunks    : {info['chunks']} (size {info['chunk_size']}, overlap {info['chunk_overlap']})")
    print(f"Vector DB : {info['vector_db']}  |  Embedding: {info['embedding']}")
    print(f"Index     : {info['index_status']}  ({time.time() - started:.1f}s)")
    if info["backend"] != "chroma":
        print(f"WARNING: ChromaDB not used -> {info['fallback_reason']}")
        sys.exit(1)
    hits = retriever.search("Can I cancel while the restaurant is preparing my food?", k=2)
    print("Sample query ->", [chunk.label for chunk, _ in hits])
