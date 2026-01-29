from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class ContextNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class ContextIncludes:
    repos: list[Path]
    chat_roots: list[Path]
    codex_session_roots: list[Path]
    note_roots: list[Path]
    repo_excludes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextIndex:
    sqlite_path: Path
    chroma_dir: Path


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    embed_model: str


@dataclass(frozen=True)
class CodexAppServerConfig:
    """Codex app-server ingestion configuration (P1.1)."""
    enabled: bool
    base_url: str
    ingest_limit: int
    timeout_sec: int


@dataclass(frozen=True)
class RankingConfig:
    """Recency decay configuration (P1.3)."""
    recency_enabled: bool
    recency_half_life_days: int


@dataclass(frozen=True)
class ArchiveConfig:
    """Archive tier configuration (P3)."""
    enabled: bool
    age_threshold_days: int
    auto_archive_on_ingest: bool
    archive_penalty: float


@dataclass(frozen=True)
class NotificationsConfig:
    """Webhook notification configuration (P3)."""
    enabled: bool
    webhook_url: str
    webhook_secret: str
    notify_on: list[str]
    min_score_for_notify: float
    retry_count: int
    retry_delay_sec: int


@dataclass(frozen=True)
class ContextConfig:
    schema_version: int
    name: str
    aliases: list[str]
    includes: ContextIncludes
    index: ContextIndex
    weights: dict[str, float]
    ollama: OllamaConfig
    created_at: str
    updated_at: str
    # P1 additions (schema v2)
    codex_appserver: CodexAppServerConfig | None = None
    ranking: RankingConfig | None = None
    state_llm: dict | None = None
    # P3 additions
    archive: ArchiveConfig | None = None
    notifications: NotificationsConfig | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ContextConfig:
        includes_data = data["includes"]
        includes = ContextIncludes(
            repos=[Path(p) for p in includes_data.get("repos", [])],
            chat_roots=[Path(p) for p in includes_data.get("chat_roots", [])],
            codex_session_roots=[Path(p) for p in includes_data.get("codex_session_roots", [])],
            note_roots=[Path(p) for p in includes_data.get("note_roots", [])],
            repo_excludes=includes_data.get("repo_excludes", []),
        )

        # Handle missing index field for old contexts
        index_data = data.get("index")
        if index_data is None:
            import os
            # Generate default index paths based on context name
            context_name = data["name"]
            indexes_root = Path(os.getenv("CHINVEX_INDEXES_ROOT", "P:/ai_memory/indexes"))
            index_root = indexes_root / context_name
            index_data = {
                "sqlite_path": str(index_root / "hybrid.db"),
                "chroma_dir": str(index_root / "chroma")
            }

        index = ContextIndex(
            sqlite_path=Path(index_data["sqlite_path"]),
            chroma_dir=Path(index_data["chroma_dir"]),
        )

        ollama_data = data.get("ollama", {})
        ollama = OllamaConfig(
            base_url=ollama_data.get("base_url", "http://skynet:11434"),
            embed_model=ollama_data.get("embed_model", "mxbai-embed-large"),
        )

        # P1: schema v2 fields (optional)
        codex_appserver = None
        if "codex_appserver" in data:
            ca_data = data["codex_appserver"]
            codex_appserver = CodexAppServerConfig(
                enabled=ca_data.get("enabled", False),
                base_url=ca_data.get("base_url", "http://localhost:8080"),
                ingest_limit=ca_data.get("ingest_limit", 100),
                timeout_sec=ca_data.get("timeout_sec", 30),
            )

        ranking = None
        if "ranking" in data:
            rank_data = data["ranking"]
            ranking = RankingConfig(
                recency_enabled=rank_data.get("recency_enabled", True),
                recency_half_life_days=rank_data.get("recency_half_life_days", 90),
            )

        state_llm = data.get("state_llm")

        # P3: archive config (optional)
        archive = None
        if "archive" in data:
            arch_data = data["archive"]
            archive = ArchiveConfig(
                enabled=arch_data.get("enabled", True),
                age_threshold_days=arch_data.get("age_threshold_days", 180),
                auto_archive_on_ingest=arch_data.get("auto_archive_on_ingest", True),
                archive_penalty=arch_data.get("archive_penalty", 0.8),
            )

        # P3: notifications config (optional)
        notifications = None
        if "notifications" in data:
            notif_data = data["notifications"]
            notifications = NotificationsConfig(
                enabled=notif_data.get("enabled", False),
                webhook_url=notif_data.get("webhook_url", ""),
                webhook_secret=notif_data.get("webhook_secret", ""),
                notify_on=notif_data.get("notify_on", ["watch_hit"]),
                min_score_for_notify=notif_data.get("min_score_for_notify", 0.75),
                retry_count=notif_data.get("retry_count", 2),
                retry_delay_sec=notif_data.get("retry_delay_sec", 5),
            )

        # Handle missing timestamp fields for old contexts
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            aliases=data.get("aliases", []),
            includes=includes,
            index=index,
            weights=data.get("weights", {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7}),
            ollama=ollama,
            created_at=data.get("created_at", now),
            updated_at=data.get("updated_at", now),
            codex_appserver=codex_appserver,
            ranking=ranking,
            state_llm=state_llm,
            archive=archive,
            notifications=notifications,
        )

    def to_dict(self) -> dict:
        result = {
            "schema_version": self.schema_version,
            "name": self.name,
            "aliases": self.aliases,
            "includes": {
                "repos": [str(p) for p in self.includes.repos],
                "chat_roots": [str(p) for p in self.includes.chat_roots],
                "codex_session_roots": [str(p) for p in self.includes.codex_session_roots],
                "note_roots": [str(p) for p in self.includes.note_roots],
                "repo_excludes": self.includes.repo_excludes,
            },
            "index": {
                "sqlite_path": str(self.index.sqlite_path),
                "chroma_dir": str(self.index.chroma_dir),
            },
            "weights": self.weights,
            "ollama": {
                "base_url": self.ollama.base_url,
                "embed_model": self.ollama.embed_model,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        # P1: schema v2 fields (optional)
        if self.codex_appserver is not None:
            result["codex_appserver"] = {
                "enabled": self.codex_appserver.enabled,
                "base_url": self.codex_appserver.base_url,
                "ingest_limit": self.codex_appserver.ingest_limit,
                "timeout_sec": self.codex_appserver.timeout_sec,
            }
        if self.ranking is not None:
            result["ranking"] = {
                "recency_enabled": self.ranking.recency_enabled,
                "recency_half_life_days": self.ranking.recency_half_life_days,
            }
        if self.state_llm is not None:
            result["state_llm"] = self.state_llm

        # P3: archive config (optional)
        if self.archive is not None:
            result["archive"] = {
                "enabled": self.archive.enabled,
                "age_threshold_days": self.archive.age_threshold_days,
                "auto_archive_on_ingest": self.archive.auto_archive_on_ingest,
                "archive_penalty": self.archive.archive_penalty,
            }

        # P3: notifications config (optional)
        if self.notifications is not None:
            result["notifications"] = {
                "enabled": self.notifications.enabled,
                "webhook_url": self.notifications.webhook_url,
                "webhook_secret": self.notifications.webhook_secret,
                "notify_on": self.notifications.notify_on,
                "min_score_for_notify": self.notifications.min_score_for_notify,
                "retry_count": self.notifications.retry_count,
                "retry_delay_sec": self.notifications.retry_delay_sec,
            }

        return result


def load_context(name: str, contexts_root: Path) -> ContextConfig:
    """Load context by name or alias with auto-upgrade from v1 to v2."""
    # Try direct name match first
    context_path = contexts_root / name / "context.json"
    if context_path.exists():
        data = json.loads(context_path.read_text(encoding="utf-8"))
        data, upgraded = _maybe_upgrade_schema(data)
        if upgraded:
            context_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return ContextConfig.from_dict(data)

    # Try alias match
    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        context_file = ctx_dir / "context.json"
        if not context_file.exists():
            continue
        data = json.loads(context_file.read_text(encoding="utf-8"))
        aliases = data.get("aliases", [])
        if name in aliases:
            data, upgraded = _maybe_upgrade_schema(data)
            if upgraded:
                context_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return ContextConfig.from_dict(data)

    raise ContextNotFoundError(
        f"Unknown context: {name}. Use 'chinvex context list' to see available contexts."
    )


def _maybe_upgrade_schema(data: dict) -> tuple[dict, bool]:
    """
    Auto-upgrade context schema from v1 to v2.

    Returns:
        (data, upgraded) tuple where upgraded is True if migration occurred
    """
    version = data.get("schema_version", 1)

    if version == 1:
        # Upgrade v1 → v2: add P1 fields with defaults
        data["schema_version"] = 2
        data.setdefault("codex_appserver", {
            "enabled": False,
            "base_url": "http://localhost:8080",
            "ingest_limit": 100,
            "timeout_sec": 30
        })
        data.setdefault("ranking", {
            "recency_enabled": True,
            "recency_half_life_days": 90
        })
        data.setdefault("state_llm", None)
        return (data, True)

    return (data, False)


def list_contexts(contexts_root: Path) -> list[ContextConfig]:
    """List all contexts, sorted by updated_at desc."""
    contexts: list[ContextConfig] = []

    if not contexts_root.exists():
        return contexts

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        context_file = ctx_dir / "context.json"
        if not context_file.exists():
            continue
        try:
            data = json.loads(context_file.read_text(encoding="utf-8"))
            contexts.append(ContextConfig.from_dict(data))
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by updated_at desc
    contexts.sort(key=lambda c: c.updated_at, reverse=True)
    return contexts
