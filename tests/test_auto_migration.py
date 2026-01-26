from pathlib import Path
import json
from chinvex.config import load_config, ConfigError
from chinvex.context import load_context


def test_auto_migrate_old_config_format(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Create old format config
    old_config = tmp_path / "old_config.json"
    old_config.write_text(json.dumps({
        "index_dir": str(tmp_path / "old_index"),
        "ollama_host": "http://127.0.0.1:11434",
        "embedding_model": "mxbai-embed-large",
        "sources": [
            {"type": "repo", "name": "myrepo", "path": "C:\\Code\\myrepo"},
            {"type": "chat", "project": "MyProject", "path": "C:\\chats"}
        ]
    }), encoding="utf-8")

    # Load should auto-migrate
    from chinvex.config import migrate_old_config
    context_name = migrate_old_config(old_config)

    # Verify migration
    assert context_name is not None
    ctx = load_context(context_name, contexts_root)
    assert len(ctx.includes.repos) == 1
    assert len(ctx.includes.chat_roots) == 1

    # Verify MIGRATED_FROM marker
    migrated_marker = (contexts_root / context_name / "MIGRATED_FROM.json")
    assert migrated_marker.exists()
    marker_data = json.loads(migrated_marker.read_text())
    assert marker_data["old_config_path"] == str(old_config)


def test_migrate_sets_context_name_from_first_repo(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    old_config = tmp_path / "config.json"
    old_config.write_text(json.dumps({
        "index_dir": str(tmp_path / "index"),
        "ollama_host": "http://127.0.0.1:11434",
        "embedding_model": "mxbai-embed-large",
        "sources": [
            {"type": "repo", "name": "coolproject", "path": "C:\\Code\\coolproject"}
        ]
    }), encoding="utf-8")

    from chinvex.config import migrate_old_config
    context_name = migrate_old_config(old_config)

    assert context_name == "coolproject"
