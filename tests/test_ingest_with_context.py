# tests/test_ingest_with_context.py
from pathlib import Path
import json
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_ingest_with_context_name(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Mock ollama
    class FakeEmbedder:
        def __init__(self, host: str, model: str, fallback_host: str | None = None):
            self.host = host
            self.model = model
            self.fallback_host = fallback_host

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    # Create context
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "test.txt").write_text("test content", encoding="utf-8")

    result = runner.invoke(app, ["context", "create", "TestCtx"])
    assert result.exit_code == 0

    # Update context to include repo
    ctx_file = contexts_root / "TestCtx" / "context.json"
    data = json.loads(ctx_file.read_text())
    data["includes"]["repos"] = [str(repo)]
    ctx_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Ingest with context name
    result = runner.invoke(app, ["ingest", "--context", "TestCtx"])
    assert result.exit_code == 0
    assert "documents: 1" in result.stdout or "TestCtx" in result.stdout


def test_ingest_requires_context_or_config(tmp_path: Path) -> None:
    # No context or config provided
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 2
    assert "--context" in result.stdout or "required" in result.stdout.lower()
