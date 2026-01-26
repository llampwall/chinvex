from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    type: str
    path: Path
    name: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class AppConfig:
    index_dir: Path
    ollama_host: str
    embedding_model: str
    sources: tuple[SourceConfig, ...]


class ConfigError(ValueError):
    pass


def _expect_str(data: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config '{key}' must be a non-empty string.")
    return value.strip()


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a JSON object.")

    index_dir = _expect_str(raw, "index_dir")
    ollama_host = _expect_str(raw, "ollama_host")
    embedding_model = _expect_str(raw, "embedding_model")
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ConfigError("Config must include a non-empty 'sources' array.")

    sources: list[SourceConfig] = []
    for i, entry in enumerate(sources_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"Source entry {i} must be an object.")
        src_type = _expect_str(entry, "type")
        if src_type not in {"repo", "chat"}:
            raise ConfigError(f"Source entry {i} has invalid type '{src_type}'.")
        path_str = _expect_str(entry, "path")
        name = _expect_str(entry, "name", required=False)
        project = _expect_str(entry, "project", required=False)
        if src_type == "repo" and not name:
            raise ConfigError(f"Source entry {i} of type 'repo' requires 'name'.")
        if src_type == "chat" and not project:
            raise ConfigError(f"Source entry {i} of type 'chat' requires 'project'.")
        sources.append(
            SourceConfig(
                type=src_type,
                path=Path(path_str),
                name=name,
                project=project,
            )
        )

    return AppConfig(
        index_dir=Path(index_dir),
        ollama_host=ollama_host,
        embedding_model=embedding_model,
        sources=tuple(sources),
    )


def migrate_old_config(old_config_path: Path) -> str:
    """
    Auto-migrate old config format to new context registry.
    Returns the created context name.
    """
    from .context_cli import get_contexts_root, get_indexes_root
    from .context import ContextConfig
    from .storage import Storage
    from .vectors import VectorStore
    from datetime import datetime, timezone

    old_cfg = load_config(old_config_path)

    # Determine context name from first repo or generate one
    context_name = None
    for src in old_cfg.sources:
        if src.type == "repo" and src.name:
            context_name = src.name
            break

    if not context_name:
        context_name = "migrated"

    contexts_root = get_contexts_root()
    indexes_root = get_indexes_root()

    ctx_dir = contexts_root / context_name
    if ctx_dir.exists():
        # Append timestamp to avoid collision
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        context_name = f"{context_name}_{timestamp}"
        ctx_dir = contexts_root / context_name

    ctx_dir.mkdir(parents=True, exist_ok=True)
    index_dir = indexes_root / context_name
    index_dir.mkdir(parents=True, exist_ok=True)

    # Build includes from old sources
    repos = []
    chat_roots = []
    for src in old_cfg.sources:
        if src.type == "repo":
            repos.append(str(src.path))
        elif src.type == "chat":
            chat_roots.append(str(src.path))

    now = datetime.now(timezone.utc).isoformat()
    context_data = {
        "schema_version": 1,
        "name": context_name,
        "aliases": [],
        "includes": {
            "repos": repos,
            "chat_roots": chat_roots,
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": str(index_dir / "hybrid.db"),
            "chroma_dir": str(index_dir / "chroma")
        },
        "weights": {
            "repo": 1.0,
            "chat": 0.8,
            "codex_session": 0.9,
            "note": 0.7
        },
        "created_at": now,
        "updated_at": now
    }

    context_file = ctx_dir / "context.json"
    context_file.write_text(json.dumps(context_data, indent=2), encoding="utf-8")

    # Write MIGRATED_FROM marker
    migrated_marker = ctx_dir / "MIGRATED_FROM.json"
    migrated_marker.write_text(
        json.dumps({
            "old_config_path": str(old_config_path),
            "migrated_at": now
        }, indent=2),
        encoding="utf-8"
    )

    # Initialize DB and Chroma
    db_path = index_dir / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    chroma_dir = index_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    VectorStore(chroma_dir)

    print(f"Auto-migrated old config to context '{context_name}'")
    print(f"Old config is now ignored. Edit {context_file} to manage sources.")

    return context_name
