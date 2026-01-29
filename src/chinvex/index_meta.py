from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexMeta:
    """Index metadata tracking embedding provider and dimensions."""
    schema_version: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    created_at: str

    def matches_provider(
        self,
        provider: str,
        model: str,
        dimensions: int
    ) -> bool:
        """Check if provider/model/dims match this index."""
        return (
            self.embedding_provider == provider and
            self.embedding_model == model and
            self.embedding_dimensions == dimensions
        )


def read_index_meta(path: Path) -> IndexMeta | None:
    """Read index metadata from meta.json, or None if missing."""
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    return IndexMeta(
        schema_version=data["schema_version"],
        embedding_provider=data["embedding_provider"],
        embedding_model=data["embedding_model"],
        embedding_dimensions=data["embedding_dimensions"],
        created_at=data["created_at"]
    )


def write_index_meta(path: Path, meta: IndexMeta) -> None:
    """Write index metadata to meta.json."""
    data = {
        "schema_version": meta.schema_version,
        "embedding_provider": meta.embedding_provider,
        "embedding_model": meta.embedding_model,
        "embedding_dimensions": meta.embedding_dimensions,
        "created_at": meta.created_at
    }
    path.write_text(json.dumps(data, indent=2))
