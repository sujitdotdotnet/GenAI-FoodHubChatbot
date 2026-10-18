"""
Policy RAG pipeline for the FoodHub chatbot.

Stages
------
1. Loading     - read the Markdown policy documents in `policies/`.
2. Chunking    - split each document by headings (##, ###), then split long sections into
                 overlapping, sentence-aligned chunks. Every chunk keeps metadata
                 (source file, document title, section, part number).
3. Embedding   - turn each chunk into a vector with a local embedding model
                 (Chroma's default all-MiniLM-L6-v2 running on ONNX Runtime; no API key needed).
4. Vector DB   - store the vectors in ChromaDB (persistent, cosine distance).
5. Indexing    - the index is built once and re-used; a content hash detects changed policies
                 and rebuilds only when needed.
6. Retrieval   - embed the question, run a similarity search and return the top-k chunks.
   (Generation happens in the app: retrieved chunks are added to the LLM prompt.)

If ChromaDB or its embedding model is unavailable, the retriever falls back to a pure-Python
TF-IDF index so the app keeps working. `PolicyRetriever.describe()` reports which backend is active.
"""

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CHUNK_SIZE = 800        # target maximum characters per chunk
CHUNK_OVERLAP = 120     # characters repeated between neighbouring chunks of one section
COLLECTION_NAME = "foodhub_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
@dataclass
class PolicyDocument:
    source: str  # file name, e.g. "cancellation_policy.md"
    title: str   # first "# " heading
    text: str


def load_policy_documents(policy_dir) -> List[PolicyDocument]:
    docs: List[PolicyDocument] = []
    folder = Path(policy_dir)
    if not folder.is_dir():
        return docs
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^#\s+(.+)$", text)
        title = match.group(1).strip() if match else path.stem.replace("_", " ").title()
        docs.append(PolicyDocument(path.name, title, text))
    return docs


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
@dataclass
class PolicyChunk:
    chunk_id: str
    source: str
    title: str
    section: str  # e.g. 'Cancelling by order status › Status "preparing food"'
    part: int     # 1-based part number when a long section is split
    parts: int    # total parts of that section
    text: str

    @property
    def label(self) -> str:
        suffix = f" (part {self.part}/{self.parts})" if self.parts > 1 else ""
        return f"{self.source} › {self.section}{suffix}"

    @property
    def embed_text(self) -> str:
        # Prefix the document title and section so the vector carries its context
        return f"{self.title} › {self.section}\n{self.text}"


def split_sections(doc: PolicyDocument) -> List[Tuple[str, str, str]]:
    """Split a Markdown document into (h2 heading, h3 heading, body) parts."""
    sections: List[Tuple[str, str, str]] = []
    h2 = h3 = ""
    buf: List[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((h2, h3, body))
        buf.clear()

    for line in doc.text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            flush()
            if level == 2:
                h2, h3 = heading, ""
            elif level == 3:
                h3 = heading
            continue
        buf.append(line)
    flush()
    return sections


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Pack paragraphs (then sentences) into chunks of at most ~chunk_size characters, with overlap."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    units: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            units.append(para)
        else:  # long paragraph -> sentences / list items
            units += [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n(?=[-*\d])", para) if s.strip()]

    chunks: List[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else ""  # start on a word boundary
            current = f"{tail} {unit}".strip()
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    docs: List[PolicyDocument], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[PolicyChunk]:
    """
    Heading-aware chunking: sub-sections (###) of one section (##) are packed together up to
    `chunk_size` characters; a sub-section that is too long on its own is split into
    overlapping sentence-aligned pieces. Sub-section headings stay inside the chunk text.
    """
    chunks: List[PolicyChunk] = []
    for doc in docs:
        counter = 0

        def emit(h2: str, h3_names: List[str], text: str, part: int = 1, parts: int = 1):
            nonlocal counter
            counter += 1
            names = " / ".join(h3_names)
            section = " › ".join(x for x in (h2 or "Overview", names) if x)
            chunks.append(
                PolicyChunk(f"{Path(doc.source).stem}-{counter:03d}", doc.source, doc.title, section, part, parts, text)
            )

        # group the (h2, h3, body) parts by their h2 heading, keeping document order
        groups: List[Tuple[str, List[Tuple[str, str]]]] = []
        for h2, h3, body in split_sections(doc):
            if not groups or groups[-1][0] != h2:
                groups.append((h2, []))
            groups[-1][1].append((h3, body))

        for h2, parts_list in groups:
            cur_text, cur_names = "", []
            for h3, body in parts_list:
                piece = f"{h3}\n{body}" if h3 else body
                if len(piece) > chunk_size:  # too long on its own -> flush, then split with overlap
                    if cur_text:
                        emit(h2, cur_names, cur_text)
                        cur_text, cur_names = "", []
                    pieces = split_text(body, chunk_size, overlap)
                    for i, sub in enumerate(pieces, start=1):
                        emit(h2, [h3] if h3 else [], f"{h3}\n{sub}" if h3 else sub, i, len(pieces))
                elif cur_text and len(cur_text) + 2 + len(piece) > chunk_size:
                    emit(h2, cur_names, cur_text)
                    cur_text, cur_names = piece, [h3] if h3 else []
                else:
                    cur_text = f"{cur_text}\n\n{piece}" if cur_text else piece
                    if h3:
                        cur_names.append(h3)
            if cur_text:
                emit(h2, cur_names, cur_text)
    return chunks


def corpus_hash(chunks: List[PolicyChunk], chunk_size: int, overlap: int) -> str:
    """Fingerprint of the indexed content; a change means the index must be rebuilt."""
    h = hashlib.sha256()
    h.update(f"{EMBEDDING_MODEL}|{chunk_size}|{overlap}".encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.embed_text.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fallback: pure-Python TF-IDF index
# ---------------------------------------------------------------------------
_STOPWORDS = set(
    """a an the and or of to in on at by for with from as is are was were be been it its this that
    i me my we our you your can could do does did will would should may might have has had not no
    if then than so but about into out up down over after before please want need get
    what where when which who how why there here order orders""".split()
)

# Common customer wording -> the wording used in the policy documents
_SYNONYMS = {
    "late": "delay", "later": "delay", "delayed": "delay", "delays": "delay", "slow": "delay",
    "cancelled": "cancel", "canceled": "cancel", "cancelling": "cancel", "canceling": "cancel",
    "cancellation": "cancel", "cancellations": "cancel",
    "charged": "charge", "charging": "charge", "deducted": "charge",
    "money": "payment", "paid": "payment", "pay": "payment",
    "reimbursement": "refund", "reimburse": "refund",
}


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [_SYNONYMS.get(w, w) for w in words]
    return [_stem(w) for w in words if w not in _STOPWORDS]


class _TfidfIndex:
    def __init__(self, docs: List[str]):
        self.doc_tokens = [_tokens(d) for d in docs]
        n = len(docs)
        df: Counter = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.doc_vecs = [self._vector(toks) for toks in self.doc_tokens]

    def _vector(self, toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def scores(self, query: str) -> List[float]:
        q = self._vector(_tokens(query))
        return [sum(w * d.get(t, 0.0) for t, w in q.items()) for d in self.doc_vecs]


# ---------------------------------------------------------------------------
# 3-6. Embedding, vector DB, indexing, retrieval
# ---------------------------------------------------------------------------
class PolicyRetriever:
    def __init__(
        self,
        chunks: List[PolicyChunk],
        n_documents: int = 0,
        persist_dir: Optional[Path] = None,
        prefer_chroma: bool = True,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunks = chunks
        self.n_documents = n_documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._by_id = {c.chunk_id: c for c in chunks}

        self.backend = "tfidf"
        self.min_score = 0.08
        self.index_status = "built in memory"
        self.fallback_reason = ""
        self._collection = None
        self._embed_fn = None
        self._tfidf = None

        if prefer_chroma and chunks:
            try:
                self._build_chroma()
                self.backend = "chroma"
                self.min_score = 0.25
            except Exception as exc:  # missing package, model download blocked, read-only disk...
                self._collection = None
                self._embed_fn = None
                self.fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
        elif not prefer_chroma:
            self.fallback_reason = "ChromaDB disabled"
        else:
            self.fallback_reason = "no policy chunks"

        if self.backend == "tfidf":
            self._tfidf = _TfidfIndex([c.embed_text for c in chunks])

    # -- construction ------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        policy_dir,
        persist_dir=None,
        prefer_chroma: bool = True,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> "PolicyRetriever":
        docs = load_policy_documents(policy_dir)
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        return cls(chunks, len(docs), persist_dir, prefer_chroma, chunk_size, chunk_overlap)

    # -- embedding ---------------------------------------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vec] for vec in self._embed_fn(texts)]

    # -- vector DB + indexing ---------------------------------------------
    def _build_chroma(self):
        import chromadb  # noqa: WPS433 (optional dependency)
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._embed_fn = DefaultEmbeddingFunction()  # downloads the ONNX model once, then cached

        client = None
        persistent = False
        if self.persist_dir is not None:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.persist_dir))
                persistent = True
            except Exception:
                client = None
        if client is None:
            client = chromadb.Client()  # in-memory
        name = COLLECTION_NAME if persistent else f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}"

        content_hash = corpus_hash(self.chunks, self.chunk_size, self.chunk_overlap)

        existing = None
        try:
            existing = client.get_collection(name, embedding_function=self._embed_fn)
        except Exception:
            existing = None

        if (
            existing is not None
            and (existing.metadata or {}).get("content_hash") == content_hash
            and existing.count() == len(self.chunks)
        ):
            self._collection = existing
            self.index_status = "loaded from disk (up to date)"
            return

        if existing is not None:  # policies changed -> rebuild
            client.delete_collection(name)

        try:
            collection = client.create_collection(
                name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"content_hash": content_hash},
                embedding_function=self._embed_fn,
            )
        except Exception:  # older ChromaDB versions
            collection = client.create_collection(
                name,
                metadata={"hnsw:space": "cosine", "content_hash": content_hash},
                embedding_function=self._embed_fn,
            )

        batch = 64
        for start in range(0, len(self.chunks), batch):
            part = self.chunks[start : start + batch]
            texts = [c.embed_text for c in part]
            collection.add(
                ids=[c.chunk_id for c in part],
                documents=texts,
                embeddings=self._embed(texts),
                metadatas=[
                    {"source": c.source, "title": c.title, "section": c.section, "part": c.part}
                    for c in part
                ],
            )
        self._collection = collection
        self.index_status = "built" + (" and saved to disk" if persistent else " in memory")

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, k: int = 4) -> List[Tuple[PolicyChunk, float]]:
        """Return up to k (chunk, similarity) pairs, best first, dropping weak matches."""
        if not self.chunks or not query.strip():
            return []

        if self.backend == "chroma":
            result = self._collection.query(
                query_embeddings=self._embed([query]),
                n_results=min(k, len(self.chunks)),
            )
            ids = result["ids"][0]
            distances = result["distances"][0]  # cosine distance = 1 - cosine similarity
            scored = [(self._by_id[i], 1.0 - float(d)) for i, d in zip(ids, distances)]
        else:
            scores = self._tfidf.scores(query)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            scored = [(self.chunks[i], scores[i]) for i in order]

        return [(c, s) for c, s in scored if s >= self.min_score]

    # -- reporting ---------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        chroma = self.backend == "chroma"
        return {
            "backend": self.backend,
            "vector_db": "ChromaDB" if chroma else "TF-IDF",
            "embedding": f"{EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS}-dim, local ONNX)" if chroma else "TF-IDF term weights",
            "similarity": "cosine similarity" if chroma else "cosine over TF-IDF vectors",
            "documents": self.n_documents,
            "chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_status": self.index_status,
            "persist_dir": str(self.persist_dir) if (chroma and self.persist_dir) else "",
            "min_score": self.min_score,
            "fallback_reason": self.fallback_reason,
        }


def format_policy_context(results: List[Tuple[PolicyChunk, float]]) -> str:
    """Numbered excerpts for the LLM prompt."""
    return "\n\n".join(
        f"[{i}] ({chunk.label})\n{chunk.text}" for i, (chunk, _) in enumerate(results, start=1)
    )
