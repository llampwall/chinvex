import json
from pathlib import Path

import pytest

from chinvex.config import AppConfig, SourceConfig
from chinvex.ingest import ingest
from chinvex_mcp.server import build_state, handle_get_chunk, handle_search


class FakeEmbedder:
    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.1, 0.2] for _ in texts]


def _write_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello world from repo", encoding="utf-8")
    return repo


def _write_chat(tmp_path: Path) -> Path:
    chat_dir = tmp_path / "chats"
    chat_dir.mkdir()
    payload = {
        "conversation_id": "conv-1",
        "title": "Test Chat",
        "project": "Twitch",
        "exported_at": "2024-01-01T00:00:00Z",
        "url": "http://example",
        "messages": [
            {"role": "user", "text": "hello chat", "timestamp": None},
            {"role": "assistant", "text": "reply here", "timestamp": None},
        ],
    }
    (chat_dir / "chat.json").write_text(json.dumps(payload), encoding="utf-8")
    return chat_dir


def _write_config(tmp_path: Path, repo: Path, chat_dir: Path) -> Path:
    config = {
        "index_dir": str(tmp_path / "index"),
        "ollama_host": "http://127.0.0.1:11434",
        "embedding_model": "mxbai-embed-large",
        "sources": [
            {"type": "repo", "name": "streamside", "path": str(repo)},
            {"type": "chat", "project": "Twitch", "path": str(chat_dir)},
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _make_config(tmp_path: Path, repo: Path, chat_dir: Path) -> AppConfig:
    return AppConfig(
        index_dir=tmp_path / "index",
        ollama_host="http://127.0.0.1:11434",
        embedding_model="mxbai-embed-large",
        sources=(
            SourceConfig(type="repo", name="streamside", path=repo),
            SourceConfig(type="chat", project="Twitch", path=chat_dir),
        ),
    )


def test_mcp_handlers_return_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path)
    chat_dir = _write_chat(tmp_path)
    config_path = _write_config(tmp_path, repo, chat_dir)
    config = _make_config(tmp_path, repo, chat_dir)

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr("chinvex_mcp.server.OllamaEmbedder", FakeEmbedder)

    ingest(config)
    state = build_state(config_path, default_k=5, default_min_score=0.0, ollama_host_override=None)
    results = handle_search(state, query="hello", source=None, k=3, min_score=0.0, include_text=False)
    assert results
    first = results[0]
    for key in ("score", "source_type", "title", "path", "chunk_id", "doc_id", "ordinal", "snippet", "meta"):
        assert key in first
    assert "text" not in first

    results_with_text = handle_search(
        state, query="hello", source=None, k=1, min_score=0.0, include_text=True
    )
    assert "text" in results_with_text[0]

    chunk = handle_get_chunk(state, chunk_id=results[0]["chunk_id"])
    for key in ("chunk_id", "source_type", "path", "title", "text", "meta"):
        assert key in chunk
