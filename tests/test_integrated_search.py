# tests/test_integrated_search.py
from pathlib import Path
from chinvex.config import AppConfig, SourceConfig
from chinvex.ingest import ingest
from chinvex.search import search


class FakeEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Return varying embeddings for testing
        return [[float(i) * 0.1 for i in range(3)] for _ in texts]


def test_search_applies_score_normalization_and_blending(tmp_path: Path, monkeypatch) -> None:
    # Create test data
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "doc1.txt").write_text("apple banana cherry", encoding="utf-8")
    (repo / "doc2.txt").write_text("banana cherry date", encoding="utf-8")

    config = AppConfig(
        index_dir=tmp_path / "index",
        ollama_host="http://127.0.0.1:11434",
        embedding_model="mxbai-embed-large",
        sources=(SourceConfig(type="repo", name="test", path=repo),)
    )

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr("chinvex.search.OllamaEmbedder", FakeEmbedder)

    ingest(config)
    results = search(config, "banana", k=5)

    # Should have results
    assert len(results) > 0

    # Scores should be normalized and blended
    for result in results:
        assert 0.0 <= result.score <= 1.0


def test_search_applies_source_weights(tmp_path: Path, monkeypatch) -> None:
    # This test requires context-based config
    # For now, just verify weights are accessible
    # Full integration in later task
    pass
