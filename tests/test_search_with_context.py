from pathlib import Path
import json
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_search_with_context_name(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Mock ollama
    class FakeEmbedder:
        def __init__(self, host: str, model: str):
            self.host = host
            self.model = model

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("chinvex.search.OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    # Create and ingest context
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "test.txt").write_text("banana apple cherry", encoding="utf-8")

    runner.invoke(app, ["context", "create", "TestCtx"])

    ctx_file = contexts_root / "TestCtx" / "context.json"
    data = json.loads(ctx_file.read_text())
    data["includes"]["repos"] = [str(repo)]
    ctx_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    runner.invoke(app, ["ingest", "--context", "TestCtx"])

    # Search with context
    result = runner.invoke(app, ["search", "--context", "TestCtx", "banana"])
    assert result.exit_code == 0
    assert "banana" in result.stdout or "test.txt" in result.stdout
