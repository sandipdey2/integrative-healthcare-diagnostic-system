"""ChromaDB index over the clinical evidence cards (data/clinical_knowledge.csv, K1..K14).

Used by workflow.KnowledgeBase when IHDS_RETRIEVAL is "auto" (default) or "chroma". The TF-IDF keyword matcher
stays as the fallback: if chromadb is not installed, the embedding model cannot be loaded, or a query fails,
retrieval silently uses keywords and the trace says so.

Embeddings (IHDS_EMBEDDINGS):
  * "onnx" (default) - chromadb's bundled all-MiniLM-L6-v2 ONNX model, run locally by onnxruntime. Needs no key;
    the ~80 MB model is downloaded once to ~/.cache/chroma on first use.
  * "openai"         - text-embedding-3-small through OPENAI_API_BASE (e.g. Vocareum). Only possible in a
    process that has OPENAI_API_KEY; the MCP tool server deliberately does not, so there it falls back to onnx.
Client: in-memory (EphemeralClient) by default; set IHDS_CHROMA_DIR (e.g. ".chroma", gitignored) for a
PersistentClient. The collection name includes a hash of the cards and the embedding model, so an edited KB or a
different model gets a fresh index.
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd

ONNX_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "text-embedding-3-small"


def _embedding_function(kind: str):
    if kind == "openai" and os.environ.get("OPENAI_API_KEY"):
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        return OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name=OPENAI_MODEL,
                                       api_base=base), f"openai-{OPENAI_MODEL}"
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction(), f"onnx-{ONNX_MODEL}"


class ChromaIndex:
    def __init__(self, df: pd.DataFrame, embeddings: str | None = None, path: str | None = None):
        import chromadb
        from chromadb.config import Settings

        kind = (embeddings or os.environ.get("IHDS_EMBEDDINGS", "onnx")).lower()
        self.ef, self.embedding = _embedding_function(kind)
        path = path if path is not None else os.environ.get("IHDS_CHROMA_DIR")
        settings = Settings(anonymized_telemetry=False, allow_reset=False)
        self.client = chromadb.PersistentClient(path=path, settings=settings) if path else \
            chromadb.EphemeralClient(settings=settings)
        self.df = df.reset_index(drop=True)
        docs = (self.df["topic"] + ". " + self.df["fact"]).tolist()
        digest = hashlib.sha1(("\n".join(docs) + self.embedding).encode()).hexdigest()[:10]
        self.collection = self.client.get_or_create_collection(
            f"ihds-evidence-{digest}", embedding_function=self.ef, metadata={"hnsw:space": "cosine"})
        if self.collection.count() != len(docs):
            self.collection.upsert(ids=self.df["id"].tolist(), documents=docs,
                                   metadatas=[{"source": s} for s in self.df["source"]])
        self._row = {cid: i for i, cid in enumerate(self.df["id"])}
        self._cache: dict[tuple, list[dict]] = {}

    def search(self, query: str, k: int = 3) -> list[dict]:
        # queries are mostly built from the top-2 risk drivers (a few dozen distinct strings), so cache them:
        # an ONNX MiniLM query embedding costs ~0.1 s on a small CPU
        key = (query, int(k))
        if key not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = self._query(query, int(k))
        return [dict(e) for e in self._cache[key]]

    def _query(self, query: str, k: int) -> list[dict]:
        r = self.collection.query(query_texts=[query], n_results=min(int(k), len(self.df)))
        out = []
        for cid, dist in zip(r["ids"][0], r["distances"][0]):
            row = self.df.iloc[self._row[cid]]
            out.append({"id": row["id"], "fact": row["fact"], "source": row["source"],
                        "score": round(1.0 - float(dist), 3)})  # cosine similarity
        return out
