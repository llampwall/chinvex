from pathlib import Path
import json
from chinvex_mcp.server import handle_chinvex_answer
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig


class FakeEmbedder:
    def __init__(self, host: str, model: str, fallback_host: str | None = None):
        self.host = host
        self.model = model
        self.fallback_host = fallback_host

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_chinvex_answer_returns_evidence_pack(tmp_path: Path, monkeypatch) -> None:
    # Setup context
    ctx = ContextConfig(
        schema_version=1,
        name="TestCtx",
        aliases=[],
        includes=ContextIncludes(repos=[], chat_roots=[], codex_session_roots=[], note_roots=[]),
        index=ContextIndex(
            sqlite_path=tmp_path / "hybrid.db",
            chroma_dir=tmp_path / "chroma"
        ),
        weights={"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        ollama=OllamaConfig(base_url="http://localhost:11434", embed_model="mxbai-embed-large"),
        created_at="2026-01-26T00:00:00Z",
        updated_at="2026-01-26T00:00:00Z"
    )

    # Create minimal index
    from chinvex.storage import Storage
    from chinvex.vectors import VectorStore

    storage = Storage(ctx.index.sqlite_path)
    storage.ensure_schema()
    storage.close()

    VectorStore(ctx.index.chroma_dir)

    monkeypatch.setattr("chinvex.search.OllamaEmbedder", FakeEmbedder)

    # Save context to temp file
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir()
    (ctx_dir / "context.json").write_text(json.dumps(ctx.to_dict(), indent=2), encoding="utf-8")

    # Call chinvex_answer
    result = handle_chinvex_answer(
        query="test query",
        context_name="TestCtx",
        k=3,
        min_score=0.1,
        contexts_root=contexts_root
    )

    # Should return evidence pack structure (even if empty)
    assert "query" in result
    assert "chunks" in result
    assert "citations" in result
    assert result["query"] == "test query"
