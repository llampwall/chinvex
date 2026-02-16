from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "chinvex_chunks") -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
        max_batch_size: int = 5000,
    ) -> None:
        """
        Upsert vectors in batches to avoid Chroma's batch size limit.

        Chroma has a max batch size limit (typically ~5461). Batch larger
        upserts to stay under the limit.
        """
        total = len(ids)
        if total <= max_batch_size:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            return

        # Batch large upserts
        for i in range(0, total, max_batch_size):
            end = min(i + max_batch_size, total)
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )

    def delete(self, ids: Iterable[str], max_batch_size: int = 5000) -> None:
        """
        Delete vectors in batches to avoid Chroma's batch size limit.
        """
        ids_list = list(ids)
        if not ids_list:
            return

        total = len(ids_list)
        if total <= max_batch_size:
            self.collection.delete(ids=ids_list)
            return

        # Batch large deletes
        for i in range(0, total, max_batch_size):
            end = min(i + max_batch_size, total)
            self.collection.delete(ids=ids_list[i:end])

    def query(self, query_embeddings: list[list[float]], n_results: int, where: dict | None = None) -> dict:
        if where:
            return self.collection.query(query_embeddings=query_embeddings, n_results=n_results, where=where)
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)

    def count(self) -> int:
        return self.collection.count()

    def get_embeddings(self, ids: list[str]) -> dict:
        """
        Get embeddings for specific chunk IDs.

        Returns dict with 'ids' and 'embeddings' keys.
        Empty lists if IDs not found.
        """
        if not ids:
            return {"ids": [], "embeddings": []}
        result = self.collection.get(ids=ids, include=["embeddings"])
        return result

    def close(self) -> None:
        """
        Close the ChromaDB client and release resources.

        Should be called before application shutdown to ensure clean
        connection termination, especially important on Windows.
        """
        if self.client is None:
            return  # Already closed

        try:
            # Stop the ChromaDB system to close SQLite connections
            if hasattr(self.client, '_system') and self.client._system is not None:
                self.client._system.stop()
        except Exception:
            # Ignore errors during cleanup
            pass

        # Clear references to allow garbage collection
        self.collection = None
        self.client = None
