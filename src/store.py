from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        meta = dict(doc.metadata)
        meta["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": meta,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_vec = self._embedding_fn(query)
        scored = []
        for record in records:
            score = _dot(query_vec, record["embedding"])
            scored.append({
                "content": record["content"],
                "score": score,
                "metadata": record["metadata"],
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids, documents, embeddings, metadatas = [], [], [], []
            for doc in docs:
                embedding = self._embedding_fn(doc.content)
                meta = dict(doc.metadata)
                meta["doc_id"] = doc.id
                unique_id = f"{doc.id}__{self._next_index}"
                self._next_index += 1
                ids.append(unique_id)
                documents.append(doc.content)
                embeddings.append(embedding)
                metadatas.append(meta)
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_vec = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            results = self._collection.query(query_embeddings=[query_vec], n_results=n)
            output = []
            for i, content in enumerate(results["documents"][0]):
                output.append({
                    "content": content,
                    "score": float(results["distances"][0][i]) if results.get("distances") else 0.0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
            return output
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k)

        if self._use_chroma and self._collection is not None:
            # Build ChromaDB where filter
            where = {k: {"$eq": v} for k, v in metadata_filter.items()}
            if len(where) == 1:
                where = {k: v["$eq"] for k, v in where.items()}
            query_vec = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            results = self._collection.query(query_embeddings=[query_vec], n_results=n, where=where)
            output = []
            for i, content in enumerate(results["documents"][0]):
                output.append({
                    "content": content,
                    "score": float(results["distances"][0][i]) if results.get("distances") else 0.0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
            return output

        filtered = [
            record for record in self._store
            if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            results = self._collection.get(where={"doc_id": doc_id})
            ids = results.get("ids", [])
            if not ids:
                return False
            self._collection.delete(ids=ids)
            return True

        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
