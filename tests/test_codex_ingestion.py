from pathlib import Path
import json
from chinvex.ingest import _ingest_codex_sessions_from_context
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig
from chinvex.storage import Storage
from chinvex.vectors import VectorStore


class FakeEmbedder:
    model = "test-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeAppServerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def list_threads(self) -> list[dict]:
        return [
            {"id": "thread-1", "title": "Test 1", "created_at": "2026-01-26T10:00:00Z", "updated_at": "2026-01-26T10:30:00Z"},
        ]

    def get_thread(self, thread_id: str) -> dict:
        return {
            "id": thread_id,
            "title": "Test Thread",
            "created_at": "2026-01-26T10:00:00Z",
            "updated_at": "2026-01-26T10:30:00Z",
            "turns": [
                {"turn_id": "turn-1", "ts": "2026-01-26T10:00:00Z", "role": "user", "text": "hello"}
            ],
            "links": {}
        }


def test_ingest_codex_sessions_with_fingerprinting(tmp_path: Path, monkeypatch) -> None:
    # Create test context
    ctx = ContextConfig(
        schema_version=1,
        name="TestCtx",
        aliases=[],
        includes=ContextIncludes(repos=[], chat_roots=[], codex_session_roots=[], note_roots=[]),
        index=ContextIndex(
            sqlite_path=tmp_path / "hybrid.db",
            chroma_dir=tmp_path / "chroma"
        ),
        weights={"codex_session": 0.9, "repo": 1.0, "chat": 0.8, "note": 0.7},
        ollama=OllamaConfig(base_url="http://localhost:11434", embed_model="mxbai-embed-large"),
        created_at="2026-01-26T00:00:00Z",
        updated_at="2026-01-26T00:00:00Z"
    )

    storage = Storage(tmp_path / "hybrid.db")
    storage.ensure_schema()

    vectors = VectorStore(tmp_path / "chroma")
    embedder = FakeEmbedder()

    monkeypatch.setattr("chinvex.ingest.AppServerClient", FakeAppServerClient)

    stats = {"documents": 0, "chunks": 0, "skipped": 0}

    _ingest_codex_sessions_from_context(
        ctx,
        "http://localhost:8080",
        storage,
        embedder,
        vectors,
        stats
    )

    # Should have ingested the thread
    assert stats["documents"] > 0

    # Verify fingerprint was created
    fp = storage.get_fingerprint("thread-1", ctx.name)
    assert fp is not None
    assert fp["source_type"] == "codex_session"
    assert fp["last_status"] == "ok"

    # Second ingest should skip
    stats2 = {"documents": 0, "chunks": 0, "skipped": 0}
    _ingest_codex_sessions_from_context(
        ctx,
        "http://localhost:8080",
        storage,
        embedder,
        vectors,
        stats2
    )

    assert stats2["skipped"] > 0
    assert stats2["documents"] == 0
