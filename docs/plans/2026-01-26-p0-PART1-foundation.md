# P0 Implementation Plan: Context Registry & Codex Ingestion - PART 1

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Foundation tasks - schema, fingerprinting, context registry, adapters, chunking

**Architecture:** Schema versioning, fingerprint tracking, context registry system, app-server adapter, conversation chunking

**Tech Stack:** Python 3.12, SQLite FTS5, Chroma, Ollama embeddings, Typer CLI, Pydantic schemas, MCP stdio protocol

---

## ✅ STATUS SUMMARY - PART 1

| Task | Status | Description |
|------|--------|-------------|
| Task 1 | ✅ DONE | Schema Version + Meta Table |
| Task 2 | ✅ DONE | Add source_fingerprints Table |
| Task 3 | ✅ DONE | Context Registry Data Structures |
| Task 4 | ✅ DONE | CLI Command - context create |
| Task 5 | ✅ DONE | CLI Command - context list |
| Task 6 | ✅ DONE | Auto-Migration from Old Config Format |
| Task 7 | ✅ DONE | Conversation Chunking with Token Approximation |
| Task 8 | ✅ DONE | Codex App-Server Client (Schema Capture) |
| Task 9 | ✅ DONE | Codex App-Server Schemas (Pydantic) |
| Task 10 | ✅ DONE | Normalize App-Server to ConversationDoc |

**Next:** PART 1 COMPLETE - Move to PART 2

---


> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement context registry system, migrate from flat config, add Codex session ingestion via app-server, implement score normalization with weight renormalization, and expose `chinvex_answer` MCP tool.

**Architecture:** Move from single JSON config to registry-based contexts (`P:\ai_memory\contexts\<Name>\context.json`). Add source fingerprinting for incremental ingest. Implement app-server adapter for Codex sessions with schema capture. Upgrade retrieval to use normalized score blending with weight renormalization. Add evidence-pack-only MCP tool.

**Tech Stack:** Python 3.12, SQLite FTS5, Chroma, Ollama embeddings, Typer CLI, Pydantic schemas, MCP stdio protocol

---

## Task 1: ✅ DONE - Schema Version + Meta Table

**Files:**
- Modify: `src/chinvex/storage.py:44-92`
- Test: `tests/test_schema_version.py`

**Step 1: Write the failing test**

```python
# tests/test_schema_version.py
from pathlib import Path
import pytest
from chinvex.storage import Storage, SCHEMA_VERSION


def test_schema_has_version_in_meta_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    cur = storage.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    assert row is not None
    assert int(row["value"]) == SCHEMA_VERSION


def test_mismatched_schema_version_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Manually set wrong version
    storage.conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    storage.conn.commit()
    storage.close()

    # Reopen should error
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        storage2 = Storage(db_path)
        storage2.ensure_schema()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema_version.py -v`
Expected: FAIL with "ImportError: cannot import name 'SCHEMA_VERSION'"

**Step 3: Implement schema versioning**

In `src/chinvex/storage.py`:

```python
# Add near top of file after imports
SCHEMA_VERSION = 1

# Modify Storage.ensure_schema() method:
def ensure_schema(self) -> None:
    self._check_fts5()

    # Create meta table first
    self._execute(
        """
        CREATE TABLE IF NOT EXISTS meta(
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )

    # Check schema version
    cur = self._execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    if row is None:
        # First time setup
        self._execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),)
        )
    else:
        stored_version = int(row["value"])
        if stored_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch: database has version {stored_version}, "
                f"but code expects version {SCHEMA_VERSION}. "
                f"Delete index folder or run migration script."
            )

    # Continue with existing table creation...
    self._execute(
        """
        CREATE TABLE IF NOT EXISTS documents(
          doc_id TEXT PRIMARY KEY,
          source_type TEXT NOT NULL,
          source_uri TEXT NOT NULL,
          project TEXT,
          repo TEXT,
          title TEXT,
          updated_at TEXT,
          content_hash TEXT,
          meta_json TEXT
        )
        """
    )
    # ... rest of existing schema
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema_version.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/storage.py tests/test_schema_version.py
git commit -m "feat: add schema versioning with meta table

- Add SCHEMA_VERSION constant
- Create meta table in ensure_schema
- Check version on open, error on mismatch
- Test version checking and mismatch handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: ✅ DONE - Add source_fingerprints Table

**Files:**
- Modify: `src/chinvex/storage.py:44-92`
- Test: `tests/test_fingerprints.py`

**Step 1: Write the failing test**

```python
# tests/test_fingerprints.py
from pathlib import Path
from chinvex.storage import Storage


def test_fingerprints_table_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    cur = storage.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='source_fingerprints'"
    )
    assert cur.fetchone() is not None


def test_upsert_fingerprint_for_file(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    storage.upsert_fingerprint(
        source_uri="C:\\Code\\test.py",
        context_name="TestContext",
        source_type="repo",
        doc_id="doc123",
        size_bytes=1024,
        mtime_unix=1234567890,
        content_sha256="abc123",
        parser_version="v1",
        chunker_version="v1",
        embedded_model="mxbai-embed-large",
        last_status="ok",
        last_error=None,
    )

    fp = storage.get_fingerprint("C:\\Code\\test.py", "TestContext")
    assert fp is not None
    assert fp["source_type"] == "repo"
    assert fp["size_bytes"] == 1024
    assert fp["last_status"] == "ok"


def test_upsert_fingerprint_for_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    storage.upsert_fingerprint(
        source_uri="thread-abc",
        context_name="TestContext",
        source_type="codex_session",
        doc_id="doc456",
        thread_updated_at="2026-01-26T10:00:00Z",
        last_turn_id="turn-999",
        parser_version="v1",
        chunker_version="v1",
        embedded_model="mxbai-embed-large",
        last_status="ok",
        last_error=None,
    )

    fp = storage.get_fingerprint("thread-abc", "TestContext")
    assert fp is not None
    assert fp["source_type"] == "codex_session"
    assert fp["thread_updated_at"] == "2026-01-26T10:00:00Z"
    assert fp["last_turn_id"] == "turn-999"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fingerprints.py -v`
Expected: FAIL with "AttributeError: 'Storage' object has no attribute 'upsert_fingerprint'"

**Step 3: Add source_fingerprints table and methods**

In `src/chinvex/storage.py`, add to `ensure_schema()`:

```python
# After chunks_fts creation
self._execute(
    """
    CREATE TABLE IF NOT EXISTS source_fingerprints (
        source_uri TEXT NOT NULL,
        context_name TEXT NOT NULL,
        source_type TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        size_bytes INTEGER,
        mtime_unix INTEGER,
        content_sha256 TEXT,
        thread_updated_at TEXT,
        last_turn_id TEXT,
        parser_version TEXT NOT NULL,
        chunker_version TEXT NOT NULL,
        embedded_model TEXT,
        last_ingested_at_unix INTEGER NOT NULL,
        last_status TEXT NOT NULL,
        last_error TEXT,
        PRIMARY KEY (source_uri, context_name)
    )
    """
)
self._execute(
    "CREATE INDEX IF NOT EXISTS idx_fingerprints_type ON source_fingerprints(source_type)"
)
self._execute(
    "CREATE INDEX IF NOT EXISTS idx_fingerprints_status ON source_fingerprints(last_status)"
)
```

Add new methods at end of Storage class:

```python
def upsert_fingerprint(
    self,
    *,
    source_uri: str,
    context_name: str,
    source_type: str,
    doc_id: str,
    parser_version: str,
    chunker_version: str,
    embedded_model: str | None,
    last_status: str,
    last_error: str | None,
    size_bytes: int | None = None,
    mtime_unix: int | None = None,
    content_sha256: str | None = None,
    thread_updated_at: str | None = None,
    last_turn_id: str | None = None,
) -> None:
    import time
    last_ingested_at_unix = int(time.time())
    self._execute(
        """
        INSERT INTO source_fingerprints(
            source_uri, context_name, source_type, doc_id,
            size_bytes, mtime_unix, content_sha256,
            thread_updated_at, last_turn_id,
            parser_version, chunker_version, embedded_model,
            last_ingested_at_unix, last_status, last_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_uri, context_name) DO UPDATE SET
            source_type=excluded.source_type,
            doc_id=excluded.doc_id,
            size_bytes=excluded.size_bytes,
            mtime_unix=excluded.mtime_unix,
            content_sha256=excluded.content_sha256,
            thread_updated_at=excluded.thread_updated_at,
            last_turn_id=excluded.last_turn_id,
            parser_version=excluded.parser_version,
            chunker_version=excluded.chunker_version,
            embedded_model=excluded.embedded_model,
            last_ingested_at_unix=excluded.last_ingested_at_unix,
            last_status=excluded.last_status,
            last_error=excluded.last_error
        """,
        (
            source_uri, context_name, source_type, doc_id,
            size_bytes, mtime_unix, content_sha256,
            thread_updated_at, last_turn_id,
            parser_version, chunker_version, embedded_model,
            last_ingested_at_unix, last_status, last_error,
        ),
    )
    self.conn.commit()

def get_fingerprint(self, source_uri: str, context_name: str) -> sqlite3.Row | None:
    cur = self._execute(
        "SELECT * FROM source_fingerprints WHERE source_uri = ? AND context_name = ?",
        (source_uri, context_name),
    )
    return cur.fetchone()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fingerprints.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/storage.py tests/test_fingerprints.py
git commit -m "feat: add source_fingerprints table for incremental ingest

- Add source_fingerprints schema with file/thread fields
- Implement upsert_fingerprint and get_fingerprint methods
- Support both file-based (mtime/size) and thread-based (updated_at/turn_id) fingerprints
- Test file and thread fingerprint storage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: ✅ DONE - Context Registry Data Structures

**Files:**
- Create: `src/chinvex/context.py`
- Test: `tests/test_context.py`

**Step 1: Write the failing test**

```python
# tests/test_context.py
from pathlib import Path
import json
import pytest
from chinvex.context import ContextConfig, load_context, list_contexts, ContextNotFoundError


def test_context_config_from_dict() -> None:
    data = {
        "schema_version": 1,
        "name": "Chinvex",
        "aliases": ["chindex"],
        "includes": {
            "repos": ["C:\\Code\\chinvex"],
            "chat_roots": ["P:\\ai_memory\\chats"],
            "codex_session_roots": ["C:\\Users\\Jordan\\.codex\\sessions"],
            "note_roots": ["P:\\ai_memory\\notes"]
        },
        "index": {
            "sqlite_path": "P:\\ai_memory\\indexes\\Chinvex\\hybrid.db",
            "chroma_dir": "P:\\ai_memory\\indexes\\Chinvex\\chroma"
        },
        "weights": {
            "repo": 1.0,
            "chat": 0.8,
            "codex_session": 0.9,
            "note": 0.7
        },
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }

    ctx = ContextConfig.from_dict(data)
    assert ctx.name == "Chinvex"
    assert "chindex" in ctx.aliases
    assert len(ctx.includes.repos) == 1
    assert ctx.weights["repo"] == 1.0


def test_load_context_by_name(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir()

    context_json = {
        "schema_version": 1,
        "name": "TestCtx",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {
            "sqlite_path": str(tmp_path / "indexes" / "TestCtx" / "hybrid.db"),
            "chroma_dir": str(tmp_path / "indexes" / "TestCtx" / "chroma")
        },
        "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }
    (ctx_dir / "context.json").write_text(json.dumps(context_json), encoding="utf-8")

    ctx = load_context("TestCtx", contexts_root)
    assert ctx.name == "TestCtx"


def test_load_context_by_alias(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    ctx_dir = contexts_root / "RealName"
    ctx_dir.mkdir()

    context_json = {
        "schema_version": 1,
        "name": "RealName",
        "aliases": ["shortname", "alias2"],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {
            "sqlite_path": str(tmp_path / "indexes" / "RealName" / "hybrid.db"),
            "chroma_dir": str(tmp_path / "indexes" / "RealName" / "chroma")
        },
        "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }
    (ctx_dir / "context.json").write_text(json.dumps(context_json), encoding="utf-8")

    ctx = load_context("shortname", contexts_root)
    assert ctx.name == "RealName"


def test_load_context_unknown_errors(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    with pytest.raises(ContextNotFoundError, match="Unknown context: NoSuchContext"):
        load_context("NoSuchContext", contexts_root)


def test_list_contexts(tmp_path: Path) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    for name in ["Alpha", "Beta"]:
        ctx_dir = contexts_root / name
        ctx_dir.mkdir()
        context_json = {
            "schema_version": 1,
            "name": name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {
                "sqlite_path": str(tmp_path / "indexes" / name / "hybrid.db"),
                "chroma_dir": str(tmp_path / "indexes" / name / "chroma")
            },
            "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
            "created_at": "2026-01-26T00:00:00Z",
            "updated_at": "2026-01-26T12:00:00Z" if name == "Beta" else "2026-01-26T08:00:00Z"
        }
        (ctx_dir / "context.json").write_text(json.dumps(context_json), encoding="utf-8")

    contexts = list_contexts(contexts_root)
    assert len(contexts) == 2
    # Should be sorted by updated_at desc
    assert contexts[0].name == "Beta"
    assert contexts[1].name == "Alpha"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.context'"

**Step 3: Implement context data structures**

```python
# src/chinvex/context.py
from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ContextIndex:
    sqlite_path: Path
    chroma_dir: Path


@dataclass(frozen=True)
class ContextConfig:
    schema_version: int
    name: str
    aliases: list[str]
    includes: ContextIncludes
    index: ContextIndex
    weights: dict[str, float]
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict) -> ContextConfig:
        includes_data = data["includes"]
        includes = ContextIncludes(
            repos=[Path(p) for p in includes_data.get("repos", [])],
            chat_roots=[Path(p) for p in includes_data.get("chat_roots", [])],
            codex_session_roots=[Path(p) for p in includes_data.get("codex_session_roots", [])],
            note_roots=[Path(p) for p in includes_data.get("note_roots", [])],
        )

        index_data = data["index"]
        index = ContextIndex(
            sqlite_path=Path(index_data["sqlite_path"]),
            chroma_dir=Path(index_data["chroma_dir"]),
        )

        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            aliases=data.get("aliases", []),
            includes=includes,
            index=index,
            weights=data["weights"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "aliases": self.aliases,
            "includes": {
                "repos": [str(p) for p in self.includes.repos],
                "chat_roots": [str(p) for p in self.includes.chat_roots],
                "codex_session_roots": [str(p) for p in self.includes.codex_session_roots],
                "note_roots": [str(p) for p in self.includes.note_roots],
            },
            "index": {
                "sqlite_path": str(self.index.sqlite_path),
                "chroma_dir": str(self.index.chroma_dir),
            },
            "weights": self.weights,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def load_context(name: str, contexts_root: Path) -> ContextConfig:
    """Load context by name or alias."""
    # Try direct name match first
    context_path = contexts_root / name / "context.json"
    if context_path.exists():
        data = json.loads(context_path.read_text(encoding="utf-8"))
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
            return ContextConfig.from_dict(data)

    raise ContextNotFoundError(
        f"Unknown context: {name}. Use 'chinvex context list' to see available contexts."
    )


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/context.py tests/test_context.py
git commit -m "feat: add context registry data structures

- Add ContextConfig, ContextIncludes, ContextIndex dataclasses
- Implement load_context (by name or alias)
- Implement list_contexts (sorted by updated_at)
- Add ContextNotFoundError for unknown contexts
- Test loading by name, alias, and listing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: ❌ TODO - CLI Command - context create

**Files:**
- Modify: `src/chinvex/cli.py`
- Create: `src/chinvex/context_cli.py`
- Test: `tests/test_context_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_context_cli.py
from pathlib import Path
import json
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_context_create_success(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, ["context", "create", "TestContext"])
    assert result.exit_code == 0
    assert "Created context: TestContext" in result.stdout

    # Verify structure
    ctx_file = contexts_root / "TestContext" / "context.json"
    assert ctx_file.exists()
    data = json.loads(ctx_file.read_text())
    assert data["name"] == "TestContext"
    assert data["schema_version"] == 1

    index_dir = indexes_root / "TestContext"
    assert index_dir.exists()
    assert (index_dir / "hybrid.db").exists()


def test_context_create_already_exists(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "TestContext"])
    result = runner.invoke(app, ["context", "create", "TestContext"])

    assert result.exit_code == 1
    assert "already exists" in result.stdout


def test_context_create_invalid_name(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, ["context", "create", "Test/Invalid"])
    assert result.exit_code == 2
    assert "invalid" in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_cli.py::test_context_create_success -v`
Expected: FAIL with "AssertionError: assert 2 == 0" (command not found)

**Step 3: Implement context create command**

```python
# src/chinvex/context_cli.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer

from .context import ContextConfig, list_contexts, load_context
from .storage import Storage
from .vectors import VectorStore


def get_contexts_root() -> Path:
    env_val = os.getenv("CHINVEX_CONTEXTS_ROOT")
    if env_val:
        return Path(env_val)
    return Path("P:/ai_memory/contexts")


def get_indexes_root() -> Path:
    env_val = os.getenv("CHINVEX_INDEXES_ROOT")
    if env_val:
        return Path(env_val)
    return Path("P:/ai_memory/indexes")


def create_context(name: str) -> None:
    """Create a new context with empty configuration."""
    # Validate name
    if not name or "/" in name or "\\" in name:
        typer.secho(f"Invalid context name: {name}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    contexts_root = get_contexts_root()
    indexes_root = get_indexes_root()

    ctx_dir = contexts_root / name
    if ctx_dir.exists():
        typer.secho(f"Context '{name}' already exists at {ctx_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Create directory structure
    ctx_dir.mkdir(parents=True, exist_ok=False)
    index_dir = indexes_root / name
    index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize context.json
    now = datetime.now(timezone.utc).isoformat()
    context_data = {
        "schema_version": 1,
        "name": name,
        "aliases": [],
        "includes": {
            "repos": [],
            "chat_roots": [],
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

    # Initialize database
    db_path = index_dir / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    # Initialize Chroma
    chroma_dir = index_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    vectors = VectorStore(chroma_dir)
    # Just instantiate to create collection

    typer.secho(f"Created context: {name}", fg=typer.colors.GREEN)
    typer.echo(f"  Config: {context_file}")
    typer.echo(f"  Index:  {index_dir}")
```

Modify `src/chinvex/cli.py`:

```python
# Add import at top
from .context_cli import create_context, get_contexts_root

# Add context subcommand group after existing commands
context_app = typer.Typer(help="Manage contexts")
app.add_typer(context_app, name="context")


@context_app.command("create")
def context_create_cmd(name: str = typer.Argument(..., help="Context name")) -> None:
    """Create a new context."""
    create_context(name)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_cli.py::test_context_create_success -v`
Expected: PASS

**Step 5: Run remaining tests**

Run: `pytest tests/test_context_cli.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/context_cli.py tests/test_context_cli.py
git commit -m "feat: add 'chinvex context create' CLI command

- Add context_cli.py with create_context function
- Support CHINVEX_CONTEXTS_ROOT and CHINVEX_INDEXES_ROOT env vars
- Initialize context.json with empty includes
- Initialize hybrid.db and Chroma collection
- Validate context name (no path separators)
- Error on duplicate context (exit 1)
- Test success, duplicate, and invalid name cases

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: ❌ TODO - CLI Command - context list

**Files:**
- Modify: `src/chinvex/context_cli.py`
- Modify: `src/chinvex/cli.py`
- Test: `tests/test_context_cli.py`

**Step 1: Write the failing test**

Add to `tests/test_context_cli.py`:

```python
def test_context_list_empty(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    result = runner.invoke(app, ["context", "list"])
    assert result.exit_code == 0
    assert "No contexts found" in result.stdout


def test_context_list_shows_contexts(tmp_path: Path, monkeypatch) -> None:
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Create two contexts
    runner.invoke(app, ["context", "create", "Alpha"])
    runner.invoke(app, ["context", "create", "Beta"])

    result = runner.invoke(app, ["context", "list"])
    assert result.exit_code == 0
    assert "Alpha" in result.stdout
    assert "Beta" in result.stdout
    assert "NAME" in result.stdout  # Header
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_cli.py::test_context_list_empty -v`
Expected: FAIL with "AssertionError" (command not found)

**Step 3: Implement context list command**

Add to `src/chinvex/context_cli.py`:

```python
def list_contexts_cli() -> None:
    """List all contexts."""
    contexts_root = get_contexts_root()

    from .context import list_contexts
    contexts = list_contexts(contexts_root)

    if not contexts:
        typer.echo("No contexts found.")
        return

    # Print table header
    typer.echo(f"{'NAME':<20} {'ALIASES':<30} {'UPDATED':<25}")
    typer.echo("-" * 75)

    for ctx in contexts:
        aliases_str = ", ".join(ctx.aliases) if ctx.aliases else "-"
        if len(aliases_str) > 28:
            aliases_str = aliases_str[:25] + "..."
        typer.echo(f"{ctx.name:<20} {aliases_str:<30} {ctx.updated_at:<25}")
```

Add to `src/chinvex/cli.py`:

```python
from .context_cli import create_context, get_contexts_root, list_contexts_cli

@context_app.command("list")
def context_list_cmd() -> None:
    """List all contexts."""
    list_contexts_cli()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_cli.py::test_context_list_empty -v`
Expected: PASS

Run: `pytest tests/test_context_cli.py::test_context_list_shows_contexts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/context_cli.py src/chinvex/cli.py tests/test_context_cli.py
git commit -m "feat: add 'chinvex context list' CLI command

- Implement list_contexts_cli function
- Display table with NAME, ALIASES, UPDATED columns
- Show 'No contexts found' when empty
- Test empty and populated lists

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: ✅ DONE - Auto-Migration from Old Config Format

**Files:**
- Modify: `src/chinvex/config.py`
- Modify: `src/chinvex/context_cli.py`
- Test: `tests/test_auto_migration.py`

**Step 1: Write the failing test**

```python
# tests/test_auto_migration.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_migration.py -v`
Expected: FAIL with "ImportError: cannot import name 'migrate_old_config'"

**Step 3: Implement auto-migration**

Add to `src/chinvex/config.py`:

```python
# Add at end of file
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_migration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/config.py tests/test_auto_migration.py
git commit -m "feat: auto-migrate old config format to context registry

- Add migrate_old_config function
- Extract context name from first repo or use 'migrated'
- Create context.json from old sources
- Write MIGRATED_FROM.json marker
- Initialize DB and Chroma for new context
- Test migration and context name extraction

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: ✅ DONE - Conversation Chunking with Token Approximation

**Files:**
- Modify: `src/chinvex/chunking.py`
- Test: `tests/test_conversation_chunking.py`

**Step 1: Write the failing test**

```python
# tests/test_conversation_chunking.py
from chinvex.chunking import chunk_conversation, approx_tokens
from math import ceil


def test_approx_tokens() -> None:
    text = "hello world"
    expected = ceil(len(text) / 4)
    assert approx_tokens(text) == expected


def test_chunk_conversation_respects_token_limit() -> None:
    # Create ConversationDoc with many long turns
    turns = []
    for i in range(50):
        turns.append({
            "turn_id": f"turn-{i}",
            "ts": "2026-01-26T10:00:00Z",
            "role": "user" if i % 2 == 0 else "assistant",
            "text": "x" * 500,  # ~125 tokens each
        })

    conversation = {
        "doc_type": "conversation",
        "source": "cx_appserver",
        "thread_id": "thread-123",
        "title": "Test Thread",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:30:00Z",
        "turns": turns,
        "links": {}
    }

    chunks = chunk_conversation(conversation, max_tokens=1500)

    # Verify chunks exist
    assert len(chunks) > 0

    # Verify no chunk exceeds token limit
    for chunk in chunks:
        tokens = approx_tokens(chunk.text)
        assert tokens <= 1500, f"Chunk exceeded token limit: {tokens} > 1500"

    # Verify turn markers present
    for chunk in chunks:
        assert "[Turn" in chunk.text


def test_chunk_conversation_never_splits_single_turn() -> None:
    # Create one giant turn that exceeds limit
    turns = [{
        "turn_id": "turn-huge",
        "ts": "2026-01-26T10:00:00Z",
        "role": "assistant",
        "text": "x" * 10000,  # ~2500 tokens, exceeds 1500 limit
    }]

    conversation = {
        "doc_type": "conversation",
        "source": "cx_appserver",
        "thread_id": "thread-huge",
        "title": "Huge Turn",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
        "turns": turns,
        "links": {}
    }

    chunks = chunk_conversation(conversation, max_tokens=1500)

    # Should have exactly 1 chunk (never split a turn)
    assert len(chunks) == 1
    # It will exceed limit, but that's OK per spec
    assert approx_tokens(chunks[0].text) > 1500
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversation_chunking.py -v`
Expected: FAIL with "ImportError: cannot import name 'chunk_conversation'"

**Step 3: Implement conversation chunking**

Add to `src/chinvex/chunking.py`:

```python
from math import ceil

MAX_CONVERSATION_TOKENS = 1500


def approx_tokens(text: str) -> int:
    """Approximate token count using len/4 heuristic."""
    return ceil(len(text) / 4)


def chunk_conversation(conversation: dict, max_tokens: int = MAX_CONVERSATION_TOKENS) -> list[Chunk]:
    """
    Chunk a ConversationDoc by logical turn groups.

    Rules:
    - Group consecutive turns up to ~max_tokens
    - Never split a single turn across chunks
    - Include [Turn N of M] markers
    - Preserve thread_id and turn_id range in metadata
    """
    turns = conversation["turns"]
    total_turns = len(turns)
    chunks: list[Chunk] = []

    if not turns:
        return [Chunk(text="", ordinal=0)]

    current_group: list[dict] = []
    current_tokens = 0
    ordinal = 0

    def flush_group():
        nonlocal ordinal, current_group, current_tokens
        if not current_group:
            return

        # Build chunk text with turn markers
        lines = []
        for turn in current_group:
            turn_idx = turns.index(turn)
            marker = f"[Turn {turn_idx + 1} of {total_turns}]"
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            lines.append(f"{marker} {role}: {text}")

        chunk_text = "\n\n".join(lines)

        # Metadata: turn range
        first_turn = current_group[0]
        last_turn = current_group[-1]

        chunks.append(
            Chunk(
                text=chunk_text,
                ordinal=ordinal,
                # Store turn_id range in metadata via constructor
                # (We'll need to update Chunk dataclass to accept arbitrary metadata)
            )
        )
        ordinal += 1
        current_group = []
        current_tokens = 0

    for turn in turns:
        turn_text = turn.get("text", "")
        turn_tokens = approx_tokens(turn_text)

        # Check if adding this turn would exceed limit
        if current_group and (current_tokens + turn_tokens > max_tokens):
            # Flush current group
            flush_group()

        # Add turn to current group
        current_group.append(turn)
        current_tokens += turn_tokens

    # Flush remaining
    flush_group()

    return chunks
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_conversation_chunking.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/chunking.py tests/test_conversation_chunking.py
git commit -m "feat: add conversation chunking with token approximation

- Add approx_tokens function (len/4 heuristic)
- Implement chunk_conversation for ConversationDoc
- Group turns up to ~1500 tokens
- Never split single turns across chunks
- Include [Turn N of M] markers in chunk text
- Test token limits and turn preservation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: ✅ DONE - Codex App-Server Client (Schema Capture)

**Files:**
- Create: `src/chinvex/adapters/__init__.py`
- Create: `src/chinvex/adapters/cx_appserver/__init__.py`
- Create: `src/chinvex/adapters/cx_appserver/client.py`
- Create: `src/chinvex/adapters/cx_appserver/capture.py`
- Test: `tests/test_appserver_client.py`

**Step 1: Write the failing test**

```python
# tests/test_appserver_client.py
from pathlib import Path
import json
import pytest
from chinvex.adapters.cx_appserver.client import AppServerClient
from chinvex.adapters.cx_appserver.capture import capture_raw_response


def test_appserver_client_list_threads(tmp_path: Path, monkeypatch) -> None:
    # Mock HTTP response
    mock_response = {
        "threads": [
            {"id": "thread-1", "title": "Test 1", "created_at": "2026-01-26T10:00:00Z", "updated_at": "2026-01-26T10:30:00Z"},
            {"id": "thread-2", "title": "Test 2", "created_at": "2026-01-25T08:00:00Z", "updated_at": "2026-01-25T08:30:00Z"}
        ]
    }

    def mock_get(url: str):
        class MockResp:
            def json(self):
                return mock_response
        return MockResp()

    monkeypatch.setattr("requests.get", mock_get)

    client = AppServerClient("http://localhost:8080")
    threads = client.list_threads()

    assert len(threads) == 2
    assert threads[0]["id"] == "thread-1"


def test_appserver_client_get_thread(tmp_path: Path, monkeypatch) -> None:
    mock_response = {
        "id": "thread-123",
        "title": "Test Thread",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:30:00Z",
        "turns": [
            {"turn_id": "turn-1", "ts": "2026-01-26T10:00:00Z", "role": "user", "text": "hello"}
        ]
    }

    def mock_get(url: str):
        class MockResp:
            def json(self):
                return mock_response
        return MockResp()

    monkeypatch.setattr("requests.get", mock_get)

    client = AppServerClient("http://localhost:8080")
    thread = client.get_thread("thread-123")

    assert thread["id"] == "thread-123"
    assert len(thread["turns"]) == 1


def test_capture_raw_response_writes_file(tmp_path: Path) -> None:
    data = {"key": "value"}
    output_dir = tmp_path / "debug" / "appserver_samples"

    filepath = capture_raw_response(data, "test_endpoint", output_dir)

    assert filepath.exists()
    content = json.loads(filepath.read_text())
    assert content["key"] == "value"
    assert "test_endpoint" in filepath.name
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_appserver_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.adapters'"

**Step 3: Implement app-server client with capture**

```python
# src/chinvex/adapters/__init__.py
# Empty file

# src/chinvex/adapters/cx_appserver/__init__.py
# Empty file

# src/chinvex/adapters/cx_appserver/client.py
from __future__ import annotations

import requests


class AppServerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def list_threads(self) -> list[dict]:
        """List all threads from /thread/list endpoint."""
        url = f"{self.base_url}/thread/list"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("threads", [])

    def get_thread(self, thread_id: str) -> dict:
        """Get full thread content from /thread/resume endpoint."""
        url = f"{self.base_url}/thread/resume/{thread_id}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()


# src/chinvex/adapters/cx_appserver/capture.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def capture_raw_response(data: dict, endpoint_name: str, output_dir: Path) -> Path:
    """
    Capture raw API response to file for schema discovery.

    Args:
        data: Raw JSON response
        endpoint_name: Name of endpoint (e.g., 'thread_list', 'thread_resume')
        output_dir: Directory to write samples (default: debug/appserver_samples/)

    Returns:
        Path to written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{endpoint_name}_{timestamp}.json"
    filepath = output_dir / filename

    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return filepath
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_appserver_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/ tests/test_appserver_client.py
git commit -m "feat: add Codex app-server client with schema capture

- Add AppServerClient with list_threads and get_thread methods
- Implement capture_raw_response for schema discovery
- Write samples to debug/appserver_samples/ with timestamps
- Test mocked HTTP calls and file capture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: ❌ TODO - Codex App-Server Schemas (Pydantic)

**Files:**
- Create: `src/chinvex/adapters/cx_appserver/schemas.py`
- Test: `tests/test_appserver_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_appserver_schemas.py
import pytest
from pydantic import ValidationError
from chinvex.adapters.cx_appserver.schemas import AppServerThread, AppServerTurn


def test_appserver_thread_schema_valid() -> None:
    data = {
        "id": "thread-123",
        "title": "Test Thread",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:30:00Z",
        "turns": []
    }

    thread = AppServerThread.model_validate(data)
    assert thread.id == "thread-123"
    assert thread.title == "Test Thread"


def test_appserver_thread_schema_missing_required_fails() -> None:
    data = {"title": "Missing ID"}

    with pytest.raises(ValidationError):
        AppServerThread.model_validate(data)


def test_appserver_turn_schema_valid() -> None:
    data = {
        "turn_id": "turn-1",
        "ts": "2026-01-26T10:00:00Z",
        "role": "user",
        "text": "hello"
    }

    turn = AppServerTurn.model_validate(data)
    assert turn.turn_id == "turn-1"
    assert turn.role == "user"
    assert turn.text == "hello"


def test_appserver_turn_with_tool() -> None:
    data = {
        "turn_id": "turn-tool",
        "ts": "2026-01-26T10:00:00Z",
        "role": "tool",
        "text": "",
        "tool": {
            "name": "bash",
            "input": {"command": "ls"},
            "output": {"stdout": "file1.txt"}
        }
    }

    turn = AppServerTurn.model_validate(data)
    assert turn.tool is not None
    assert turn.tool["name"] == "bash"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_appserver_schemas.py -v`
Expected: FAIL with "ImportError: cannot import name 'AppServerThread'"

**Step 3: Implement Pydantic schemas**

```python
# src/chinvex/adapters/cx_appserver/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class AppServerTurn(BaseModel):
    """
    Schema for a turn from app-server API.

    Discovered from actual /thread/resume responses.
    """
    turn_id: str
    ts: str  # ISO8601
    role: str  # user|assistant|tool|system
    text: str | None = None
    tool: dict | None = None
    attachments: list[dict] = Field(default_factory=list)
    meta: dict | None = None


class AppServerThread(BaseModel):
    """
    Schema for a thread from app-server API.

    Discovered from actual /thread/list and /thread/resume responses.
    """
    id: str
    title: str | None = None
    created_at: str  # ISO8601
    updated_at: str  # ISO8601
    turns: list[AppServerTurn] = Field(default_factory=list)
    links: dict = Field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_appserver_schemas.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/cx_appserver/schemas.py tests/test_appserver_schemas.py
git commit -m "feat: add Pydantic schemas for app-server responses

- Add AppServerThread schema
- Add AppServerTurn schema
- Support optional tool, attachments, meta fields
- Validate against discovered API response shapes
- Test valid schemas and missing required fields

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: ❌ TODO - Normalize App-Server to ConversationDoc

**Files:**
- Create: `src/chinvex/adapters/cx_appserver/normalize.py`
- Test: `tests/test_appserver_normalize.py`

**Step 1: Write the failing test**

```python
# tests/test_appserver_normalize.py
from chinvex.adapters.cx_appserver.normalize import normalize_to_conversation_doc
from chinvex.adapters.cx_appserver.schemas import AppServerThread, AppServerTurn


def test_normalize_appserver_thread_to_conversation_doc() -> None:
    thread = AppServerThread(
        id="thread-abc",
        title="Test Conversation",
        created_at="2026-01-26T10:00:00Z",
        updated_at="2026-01-26T10:30:00Z",
        turns=[
            AppServerTurn(
                turn_id="turn-1",
                ts="2026-01-26T10:00:00Z",
                role="user",
                text="hello"
            ),
            AppServerTurn(
                turn_id="turn-2",
                ts="2026-01-26T10:01:00Z",
                role="assistant",
                text="hi there"
            )
        ],
        links={"workspace_id": "ws-123"}
    )

    doc = normalize_to_conversation_doc(thread)

    assert doc["doc_type"] == "conversation"
    assert doc["source"] == "cx_appserver"
    assert doc["thread_id"] == "thread-abc"
    assert doc["title"] == "Test Conversation"
    assert len(doc["turns"]) == 2
    assert doc["turns"][0]["role"] == "user"
    assert doc["links"]["workspace_id"] == "ws-123"


def test_normalize_preserves_tool_info() -> None:
    thread = AppServerThread(
        id="thread-tool",
        title=None,
        created_at="2026-01-26T10:00:00Z",
        updated_at="2026-01-26T10:00:00Z",
        turns=[
            AppServerTurn(
                turn_id="turn-tool",
                ts="2026-01-26T10:00:00Z",
                role="tool",
                text="",
                tool={"name": "bash", "input": {}, "output": {}}
            )
        ]
    )

    doc = normalize_to_conversation_doc(thread)

    assert doc["turns"][0]["tool"] is not None
    assert doc["turns"][0]["tool"]["name"] == "bash"


def test_normalize_generates_title_from_first_user_message() -> None:
    thread = AppServerThread(
        id="thread-no-title",
        title=None,
        created_at="2026-01-26T10:00:00Z",
        updated_at="2026-01-26T10:00:00Z",
        turns=[
            AppServerTurn(
                turn_id="turn-1",
                ts="2026-01-26T10:00:00Z",
                role="user",
                text="This is a long user message that should be truncated to form the title"
            )
        ]
    )

    doc = normalize_to_conversation_doc(thread)

    assert doc["title"] is not None
    assert len(doc["title"]) <= 60
    assert "This is a long user message" in doc["title"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_appserver_normalize.py -v`
Expected: FAIL with "ImportError: cannot import name 'normalize_to_conversation_doc'"

**Step 3: Implement normalization**

```python
# src/chinvex/adapters/cx_appserver/normalize.py
from __future__ import annotations

from .schemas import AppServerThread


def normalize_to_conversation_doc(thread: AppServerThread) -> dict:
    """
    Normalize AppServerThread to ConversationDoc internal schema.

    Returns dict matching ConversationDoc schema from P0 spec §2.
    """
    # Generate title if missing
    title = thread.title
    if not title and thread.turns:
        # Use first user message truncated
        for turn in thread.turns:
            if turn.role == "user" and turn.text:
                title = turn.text[:60].strip()
                if len(turn.text) > 60:
                    title += "..."
                break

    if not title:
        title = f"Thread {thread.id}"

    # Normalize turns
    normalized_turns = []
    for turn in thread.turns:
        normalized_turn = {
            "turn_id": turn.turn_id,
            "ts": turn.ts,
            "role": turn.role,
            "text": turn.text or "",
        }

        if turn.tool:
            normalized_turn["tool"] = turn.tool

        if turn.attachments:
            normalized_turn["attachments"] = turn.attachments

        if turn.meta:
            normalized_turn["meta"] = turn.meta

        normalized_turns.append(normalized_turn)

    return {
        "doc_type": "conversation",
        "source": "cx_appserver",
        "thread_id": thread.id,
        "title": title,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "turns": normalized_turns,
        "links": thread.links,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_appserver_normalize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/adapters/cx_appserver/normalize.py tests/test_appserver_normalize.py
git commit -m "feat: normalize app-server threads to ConversationDoc

- Implement normalize_to_conversation_doc function
- Map AppServerThread to internal ConversationDoc schema
- Generate title from first user message if missing
- Preserve tool, attachments, meta fields in turns
- Test normalization, tool preservation, title generation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

