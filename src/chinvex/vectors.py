from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "chinvex_chunks") -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def delete(self, ids: Iterable[str]) -> None:
        ids_list = list(ids)
        if ids_list:
            self.collection.delete(ids=ids_list)

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
