# P3 Tasks 17-26 Detailed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Archive Tier, Webhook Notifications, and Gateway Extras (Tasks 17-26 from P3 plan)

**Architecture:** Flag-based archive filtering in search, webhook notifications with HMAC signatures, optional Redis rate limiting, Prometheus metrics endpoint

**Tech Stack:** SQLite (archive flags), HMAC-SHA256 (webhook signatures), Redis (optional), Prometheus client

---

## Prerequisites

- Tasks 1-16 completed (chunking v2, cross-context search, watch history, archive schema)
- Archive schema migration (v3) applied
- Watch history logging functional
- Python 3.12+ environment

---

## P3c: Policy + Ops

### Phase 4: Archive Tier (P3.4)

---

### Task 17: Archive run command (dry-run + execute)

**Files:**
- Create: `tests/test_archive_commands.py`
- Modify: `src/chinvex/cli.py`
- Create: `src/chinvex/archive.py`

**Step 1: Write failing test**

Create `tests/test_archive_commands.py`:

```python
"""Test archive commands."""
import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_archive_run_dry_run_default(tmp_path):
    """Test that archive run is dry-run by default."""
    result = runner.invoke(app, [
        "archive", "run",
        "--context", "Test",
        "--older-than", "180d"
    ])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout.lower()
    assert "would archive" in result.stdout.lower()


def test_archive_run_force_executes(tmp_path):
    """Test that --force executes archive."""
    result = runner.invoke(app, [
        "archive", "run",
        "--context", "Test",
        "--older-than", "180d",
        "--force"
    ])
    assert result.exit_code == 0
    assert "archived" in result.stdout.lower()


def test_archive_run_respects_threshold(tmp_path):
    """Test that only docs older than threshold are archived."""
    from chinvex.archive import archive_old_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert test documents
    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()
    recent_date = (now - timedelta(days=30)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, title, updated_at, ingested_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "Old Doc", old_date, old_date, 0)
    )
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, title, updated_at, ingested_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_recent", "repo", "Recent Doc", recent_date, recent_date, 0)
    )
    storage.conn.commit()

    # Run archive
    count = archive_old_documents(storage, age_threshold_days=180)

    # Verify only old doc archived
    assert count == 1

    cursor = storage.conn.execute("SELECT doc_id, archived FROM documents ORDER BY doc_id")
    rows = cursor.fetchall()
    assert rows[0]["doc_id"] == "doc_old"
    assert rows[0]["archived"] == 1
    assert rows[1]["doc_id"] == "doc_recent"
    assert rows[1]["archived"] == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_archive_commands.py::test_archive_run_dry_run_default -v`
Expected: FAIL with "no such command: archive"

**Step 3: Implement archive module**

Create `src/chinvex/archive.py`:

```python
"""Archive tier implementation."""
from datetime import datetime, timedelta
from .storage import Storage
from .util import now_iso


def get_doc_age_timestamp(row: dict) -> datetime | None:
    """
    Get the timestamp used for archive age calculation.

    Prefers updated_at (content age), falls back to ingested_at.
    """
    ts_str = row.get("updated_at") or row.get("ingested_at")
    if not ts_str:
        return None

    # Parse ISO8601 timestamp
    try:
        # Remove 'Z' suffix if present
        ts_str = ts_str.rstrip("Z")
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def archive_old_documents(storage: Storage, age_threshold_days: int, dry_run: bool = False) -> int:
    """
    Archive documents older than threshold.

    Returns count of documents archived (or would be archived in dry-run).
    """
    threshold_date = datetime.utcnow() - timedelta(days=age_threshold_days)

    # Find candidates
    cursor = storage.conn.execute(
        """
        SELECT doc_id, updated_at, ingested_at
        FROM documents
        WHERE archived = 0
        """
    )

    candidates = []
    for row in cursor.fetchall():
        row_dict = dict(row)
        doc_age = get_doc_age_timestamp(row_dict)
        if doc_age and doc_age < threshold_date:
            candidates.append(row_dict["doc_id"])

    if dry_run:
        return len(candidates)

    # Execute archive
    if candidates:
        archived_at = now_iso()
        placeholders = ",".join("?" * len(candidates))
        storage._execute(
            f"""
            UPDATE documents
            SET archived = 1, archived_at = ?
            WHERE doc_id IN ({placeholders})
            """,
            (archived_at, *candidates)
        )
        storage.conn.commit()

    return len(candidates)
```

**Step 4: Add CLI command**

Edit `src/chinvex/cli.py`, add:

```python
@app.command()
def archive(
    action: str = typer.Argument(..., help="Action: run, list, restore, purge"),
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    older_than: str | None = typer.Option(None, "--older-than", help="Age threshold (e.g., 180d)"),
    force: bool = typer.Option(False, "--force", help="Execute action (not dry-run)"),
    doc_id: str | None = typer.Option(None, "--doc-id", help="Document ID for restore"),
    limit: int = typer.Option(50, "--limit", help="Limit for list command"),
):
    """Manage archive tier."""
    from chinvex.config import load_config
    from chinvex.archive import archive_old_documents
    from chinvex.storage import Storage

    config = load_config(context)
    db_path = config.index_dir / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    if action == "run":
        # Parse threshold
        if older_than:
            if older_than.endswith('d'):
                days = int(older_than[:-1])
            else:
                raise ValueError(f"Invalid threshold format: {older_than}")
        else:
            days = config.archive.age_threshold_days if hasattr(config, 'archive') else 180

        # Run archive
        count = archive_old_documents(storage, age_threshold_days=days, dry_run=not force)

        if force:
            print(f"Archived {count} docs (older than {days}d)")
        else:
            print(f"Would archive {count} docs (dry-run)")
            print("Use --force to execute")

    elif action == "list":
        # Will implement in Task 19
        print("List not yet implemented")

    elif action == "restore":
        # Will implement in Task 19
        print("Restore not yet implemented")

    elif action == "purge":
        # Will implement in Task 21
        print("Purge not yet implemented")

    else:
        print(f"Unknown action: {action}")
        raise typer.Exit(1)
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_archive_commands.py -v`
Expected: Tests pass

**Step 6: Manual test**

Run: `chinvex archive run --context Test --older-than 180d`
Expected: Shows "Would archive N docs (dry-run)"

**Step 7: Commit**

```bash
git add src/chinvex/archive.py src/chinvex/cli.py tests/test_archive_commands.py
git commit -m "feat(archive): add archive run command

- Implement archive_old_documents() with age threshold
- Dry-run by default, --force to execute
- Use updated_at or ingested_at for age calculation
- Update documents.archived = 1 and archived_at

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 18: Search filtering with archive penalty

**Files:**
- Modify: `src/chinvex/search.py`
- Create: `tests/test_archive_search_filtering.py`

**Step 1: Write failing test**

Create `tests/test_archive_search_filtering.py`:

```python
"""Test archive filtering in search."""
import pytest
import sqlite3
from datetime import datetime, timedelta
from chinvex.search import search, SearchResult
from chinvex.config import AppConfig
from pathlib import Path


def test_search_excludes_archived_by_default(tmp_path):
    """Test that archived docs are excluded from search by default."""
    from chinvex.storage import Storage

    # Setup test database
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert archived and active docs
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, title, archived) VALUES (?, ?, ?, ?)",
        ("doc_archived", "repo", "Archived Doc", 1)
    )
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, title, archived) VALUES (?, ?, ?, ?)",
        ("doc_active", "repo", "Active Doc", 0)
    )
    storage.conn.commit()

    # TODO: Search and verify archived excluded
    # This will fail until search() is updated


def test_search_includes_archive_with_flag(tmp_path):
    """Test that include_archive flag includes archived docs."""
    # TODO: Test include_archive parameter


def test_search_applies_archive_penalty(tmp_path):
    """Test that archived docs get penalty, not recency decay."""
    # TODO: Test archive penalty multiplier
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_archive_search_filtering.py::test_search_excludes_archived_by_default -v`
Expected: FAIL (test incomplete or archived not filtered)

**Step 3: Modify search function**

Edit `src/chinvex/search.py`, locate the `search()` function and add archive filtering:

```python
def search(
    config: AppConfig,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    ollama_host_override: str | None = None,
    weights: dict[str, float] | None = None,
    include_archive: bool = False,  # NEW
) -> list[SearchResult]:
    db_path = config.index_dir / "hybrid.db"
    chroma_dir = config.index_dir / "chroma"
    storage = Storage(db_path)
    storage.ensure_schema()

    filters = {}
    if source in {"repo", "chat"}:
        filters["source_type"] = source
    if project:
        filters["project"] = project
    if repo:
        filters["repo"] = repo

    # NEW: Add archive filter
    if not include_archive:
        filters["archived"] = 0

    # ... rest of search logic ...
```

Then modify the scoring logic to apply archive penalty:

```python
def apply_score_adjustments(result: SearchResult, config: AppConfig, archived: int, include_archive: bool) -> SearchResult | None:
    """
    Apply score adjustments for recency and archive status.

    Returns None if result should be filtered out.
    """
    if archived == 1:
        if not include_archive:
            return None  # Exclude archived docs
        else:
            # Apply archive penalty (replaces recency decay)
            archive_penalty = config.archive.archive_penalty if hasattr(config, 'archive') else 0.8
            result.score *= archive_penalty
    else:
        # Active doc: apply recency decay
        # (existing recency decay logic here)
        pass

    return result
```

**Step 4: Update SQL queries**

Find the SQL queries in `search()` and add archive column:

```python
# Example: Update the chunk retrieval query
cursor = storage.conn.execute(
    """
    SELECT c.chunk_id, c.text, d.source_type, d.title, d.source_uri, d.archived
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    WHERE c.chunk_id = ?
    """,
    (chunk_id,)
)
```

**Step 5: Complete test implementation**

Update `tests/test_archive_search_filtering.py` with complete test logic.

**Step 6: Run tests**

Run: `python -m pytest tests/test_archive_search_filtering.py -v`
Expected: Tests pass

**Step 7: Commit**

```bash
git add src/chinvex/search.py tests/test_archive_search_filtering.py
git commit -m "feat(search): add archive filtering and penalty

- Filter out archived docs by default
- Add include_archive parameter to search()
- Apply archive_penalty when include_archive=True
- Archive penalty replaces recency decay for archived docs

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 19: Archive list and restore commands

**Files:**
- Modify: `src/chinvex/cli.py`
- Modify: `src/chinvex/archive.py`
- Modify: `tests/test_archive_commands.py`

**Step 1: Write failing test**

Add to `tests/test_archive_commands.py`:

```python
def test_archive_list_shows_archived_docs(tmp_path):
    """Test that archive list shows archived documents."""
    result = runner.invoke(app, [
        "archive", "list",
        "--context", "Test",
        "--limit", "50"
    ])
    assert result.exit_code == 0
    # Should show archived docs


def test_archive_restore_unarchives_doc(tmp_path):
    """Test that restore flips archived flag."""
    from chinvex.archive import restore_document
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert archived doc
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, title, archived, archived_at) VALUES (?, ?, ?, ?, ?)",
        ("doc_archived", "repo", "Archived Doc", 1, "2026-01-01T00:00:00Z")
    )
    storage.conn.commit()

    # Restore
    restore_document(storage, "doc_archived")

    # Verify unarchived
    cursor = storage.conn.execute("SELECT archived, archived_at FROM documents WHERE doc_id = ?", ("doc_archived",))
    row = cursor.fetchone()
    assert row["archived"] == 0
    assert row["archived_at"] is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_archive_commands.py::test_archive_list_shows_archived_docs -v`
Expected: FAIL with "List not yet implemented"

**Step 3: Implement list function**

Edit `src/chinvex/archive.py`, add:

```python
def list_archived_documents(storage: Storage, limit: int = 50) -> list[dict]:
    """
    List archived documents.

    Returns list of archived document metadata.
    """
    cursor = storage.conn.execute(
        """
        SELECT doc_id, source_type, title, archived_at
        FROM documents
        WHERE archived = 1
        ORDER BY archived_at DESC
        LIMIT ?
        """,
        (limit,)
    )

    return [dict(row) for row in cursor.fetchall()]


def restore_document(storage: Storage, doc_id: str) -> bool:
    """
    Restore archived document.

    Flips archived flag to 0. Does NOT re-ingest or re-embed.
    Returns True if document was found and restored.
    """
    cursor = storage.conn.execute(
        "SELECT archived FROM documents WHERE doc_id = ?",
        (doc_id,)
    )
    row = cursor.fetchone()

    if not row:
        return False

    if row["archived"] == 0:
        return False  # Already active

    storage._execute(
        "UPDATE documents SET archived = 0, archived_at = NULL WHERE doc_id = ?",
        (doc_id,)
    )
    storage.conn.commit()

    return True
```

**Step 4: Update CLI commands**

Edit `src/chinvex/cli.py`, update the `archive()` function:

```python
# In archive() function, replace list/restore stubs:

if action == "list":
    from chinvex.archive import list_archived_documents

    docs = list_archived_documents(storage, limit=limit)

    if not docs:
        print("No archived documents found")
    else:
        print(f"{'Doc ID':<40} {'Type':<10} {'Title':<40} {'Archived At':<20}")
        print("-" * 115)
        for doc in docs:
            doc_id = doc["doc_id"][:39]
            source_type = doc["source_type"][:9]
            title = doc["title"][:39]
            archived_at = doc["archived_at"][:19] if doc["archived_at"] else "N/A"
            print(f"{doc_id:<40} {source_type:<10} {title:<40} {archived_at:<20}")

elif action == "restore":
    from chinvex.archive import restore_document

    if not doc_id:
        print("Error: --doc-id required for restore")
        raise typer.Exit(1)

    success = restore_document(storage, doc_id)

    if success:
        print(f"Restored document: {doc_id}")
    else:
        print(f"Document not found or already active: {doc_id}")
        raise typer.Exit(1)
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_archive_commands.py -v`
Expected: Tests pass

**Step 6: Manual test**

Run:
```bash
chinvex archive list --context Test
chinvex archive restore --context Test --doc-id abc123
```

Expected: List shows archived docs, restore unarchives specific doc

**Step 7: Commit**

```bash
git add src/chinvex/archive.py src/chinvex/cli.py tests/test_archive_commands.py
git commit -m "feat(archive): add list and restore commands

- Add archive list to show archived documents
- Add archive restore to unarchive specific doc
- Restore flips archived flag only (no re-ingest)
- Display formatted table for list command

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 20: Auto-archive on ingest

**Files:**
- Modify: `src/chinvex/ingest.py`
- Create: `tests/test_auto_archive.py`

**Step 1: Write failing test**

Create `tests/test_auto_archive.py`:

```python
"""Test auto-archive on ingest."""
import pytest
from datetime import datetime, timedelta
from pathlib import Path


def test_auto_archive_runs_after_ingest(tmp_path):
    """Test that auto-archive runs when enabled in config."""
    from chinvex.ingest import run_ingest
    from chinvex.storage import Storage
    from chinvex.config import AppConfig

    # Setup test context with old doc
    # (This test will need proper setup of test context)
    # TODO: Complete test implementation


def test_auto_archive_respects_enabled_flag(tmp_path):
    """Test that auto-archive only runs when enabled."""
    # TODO: Test config.archive.enabled flag


def test_auto_archive_logs_count(tmp_path, capsys):
    """Test that auto-archive logs archived count."""
    # TODO: Test log message appears
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auto_archive.py::test_auto_archive_runs_after_ingest -v`
Expected: FAIL (auto-archive not implemented)

**Step 3: Add auto-archive hook to ingest**

Edit `src/chinvex/ingest.py`, locate the end of the ingest function (after all ingestion completes):

```python
def run_ingest(config: AppConfig, sources: list[str], ollama_host_override: str | None = None):
    """Run ingestion for specified sources."""
    # ... existing ingestion logic ...

    # After all ingestion completes
    storage.conn.commit()

    # NEW: Auto-archive hook
    if hasattr(config, 'archive') and config.archive.enabled and config.archive.auto_archive_on_ingest:
        from .archive import archive_old_documents

        archived_count = archive_old_documents(
            storage,
            age_threshold_days=config.archive.age_threshold_days,
            dry_run=False
        )

        if archived_count > 0:
            print(f"Archived {archived_count} docs (older than {config.archive.age_threshold_days}d)")

    # ... rest of existing code ...
```

**Step 4: Add archive config to AppConfig**

Check if `src/chinvex/config.py` needs archive config dataclass:

```python
@dataclass
class ArchiveConfig:
    enabled: bool = True
    age_threshold_days: int = 180
    auto_archive_on_ingest: bool = True
    archive_penalty: float = 0.8


@dataclass
class AppConfig:
    # ... existing fields ...
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
```

**Step 5: Complete test implementation**

Update `tests/test_auto_archive.py` with complete test logic using fixtures.

**Step 6: Run tests**

Run: `python -m pytest tests/test_auto_archive.py -v`
Expected: Tests pass

**Step 7: Manual test**

Create a test document with old timestamp, run ingest:
```bash
chinvex ingest --context Test
```
Expected: Sees "Archived N docs" message if old docs found

**Step 8: Commit**

```bash
git add src/chinvex/ingest.py src/chinvex/config.py tests/test_auto_archive.py
git commit -m "feat(ingest): add auto-archive post-ingest hook

- Run archive check after ingest when enabled
- Respect archive.auto_archive_on_ingest config
- Log archived count to console
- Add ArchiveConfig to AppConfig

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 21: Archive purge command

**Files:**
- Modify: `src/chinvex/archive.py`
- Modify: `src/chinvex/cli.py`
- Modify: `tests/test_archive_commands.py`

**Step 1: Write failing test**

Add to `tests/test_archive_commands.py`:

```python
def test_archive_purge_dry_run_default(tmp_path):
    """Test that purge is dry-run by default."""
    result = runner.invoke(app, [
        "archive", "purge",
        "--context", "Test",
        "--older-than", "365d"
    ])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout.lower()
    assert "would purge" in result.stdout.lower()


def test_archive_purge_deletes_permanently(tmp_path):
    """Test that purge --force deletes archived docs."""
    from chinvex.archive import purge_archived_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert very old archived doc
    now = datetime.utcnow()
    very_old = (now - timedelta(days=400)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, title, archived, archived_at) VALUES (?, ?, ?, ?, ?)",
        ("doc_very_old", "repo", "Very Old", 1, very_old)
    )
    storage.conn.commit()

    # Purge
    count = purge_archived_documents(storage, age_threshold_days=365, dry_run=False)

    # Verify deleted
    assert count == 1
    cursor = storage.conn.execute("SELECT COUNT(*) as count FROM documents WHERE doc_id = ?", ("doc_very_old",))
    assert cursor.fetchone()["count"] == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_archive_commands.py::test_archive_purge_dry_run_default -v`
Expected: FAIL with "Purge not yet implemented"

**Step 3: Implement purge function**

Edit `src/chinvex/archive.py`, add:

```python
def purge_archived_documents(storage: Storage, age_threshold_days: int, dry_run: bool = False) -> int:
    """
    Permanently delete archived documents older than threshold.

    Only deletes documents that are ALREADY archived.
    Returns count of documents purged.
    """
    threshold_date = datetime.utcnow() - timedelta(days=age_threshold_days)

    # Find candidates (archived docs older than threshold)
    cursor = storage.conn.execute(
        """
        SELECT doc_id, archived_at
        FROM documents
        WHERE archived = 1 AND archived_at IS NOT NULL
        """
    )

    candidates = []
    for row in cursor.fetchall():
        archived_at_str = row["archived_at"].rstrip("Z")
        try:
            archived_at = datetime.fromisoformat(archived_at_str)
            if archived_at < threshold_date:
                candidates.append(row["doc_id"])
        except ValueError:
            continue

    if dry_run:
        return len(candidates)

    # Execute purge (delete from both documents and chunks)
    if candidates:
        placeholders = ",".join("?" * len(candidates))

        # Delete chunks first (foreign key constraint)
        storage._execute(
            f"DELETE FROM chunks WHERE doc_id IN ({placeholders})",
            candidates
        )

        # Delete documents
        storage._execute(
            f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
            candidates
        )

        storage.conn.commit()

    return len(candidates)
```

**Step 4: Update CLI command**

Edit `src/chinvex/cli.py`, update the `archive()` function:

```python
# In archive() function, replace purge stub:

elif action == "purge":
    from chinvex.archive import purge_archived_documents

    if not older_than:
        print("Error: --older-than required for purge")
        raise typer.Exit(1)

    # Parse threshold
    if older_than.endswith('d'):
        days = int(older_than[:-1])
    else:
        raise ValueError(f"Invalid threshold format: {older_than}")

    # Run purge
    count = purge_archived_documents(storage, age_threshold_days=days, dry_run=not force)

    if force:
        print(f"Purged {count} docs permanently (older than {days}d)")
    else:
        print(f"Would purge {count} docs (dry-run)")
        print("Use --force to execute")
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_archive_commands.py -v`
Expected: Tests pass

**Step 6: Manual test**

Run: `chinvex archive purge --context Test --older-than 365d`
Expected: Shows "Would purge N docs (dry-run)"

**Step 7: Commit**

```bash
git add src/chinvex/archive.py src/chinvex/cli.py tests/test_archive_commands.py
git commit -m "feat(archive): add purge command

- Permanently delete archived docs older than threshold
- Dry-run by default, --force to execute
- Delete from both documents and chunks tables
- Only purges already-archived documents

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Webhooks (P3.2b)

---

### Task 22: Webhook notification implementation

**Files:**
- Create: `src/chinvex/notifications.py`
- Create: `tests/test_webhook_notifications.py`

**Step 1: Write failing test**

Create `tests/test_webhook_notifications.py`:

```python
"""Test webhook notifications."""
import pytest
import json
from unittest.mock import patch, Mock


def test_webhook_url_validation_https_required():
    """Test that HTTP URLs are rejected."""
    from chinvex.notifications import validate_webhook_url

    assert validate_webhook_url("https://example.com/webhook") is True
    assert validate_webhook_url("http://example.com/webhook") is False


def test_webhook_url_validation_blocks_private_ips():
    """Test that private IPs are blocked."""
    from chinvex.notifications import validate_webhook_url

    assert validate_webhook_url("https://127.0.0.1/webhook") is False
    assert validate_webhook_url("https://localhost/webhook") is False
    assert validate_webhook_url("https://192.168.1.1/webhook") is False
    assert validate_webhook_url("https://10.0.0.1/webhook") is False


def test_send_webhook_sanitizes_source_uri():
    """Test that source_uri is sanitized to filename only."""
    from chinvex.notifications import sanitize_source_uri

    assert sanitize_source_uri(r"C:\Users\Jordan\Private\diary.md") == "diary.md"
    assert sanitize_source_uri("/home/user/secret/file.txt") == "file.txt"
    assert sanitize_source_uri("file.txt") == "file.txt"


@patch('chinvex.notifications.requests.post')
def test_send_webhook_posts_payload(mock_post):
    """Test that send_webhook posts correct payload."""
    from chinvex.notifications import send_webhook

    mock_post.return_value = Mock(status_code=200)

    payload = {
        "event": "watch_hit",
        "watch_id": "test_watch",
        "query": "test query",
        "hits": [{"chunk_id": "abc", "score": 0.85, "snippet": "test"}]
    }

    success = send_webhook("https://example.com/webhook", payload)

    assert success is True
    mock_post.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webhook_notifications.py::test_webhook_url_validation_https_required -v`
Expected: FAIL with "module not found"

**Step 3: Implement notifications module**

Create `src/chinvex/notifications.py`:

```python
"""Webhook notification implementation."""
import ipaddress
import socket
from urllib.parse import urlparse
from pathlib import Path
import time


def validate_webhook_url(url: str) -> bool:
    """
    Validate webhook URL for security.

    Requirements:
    - HTTPS only
    - No private IPs (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - No localhost
    """
    try:
        parsed = urlparse(url)

        # HTTPS required
        if parsed.scheme != 'https':
            return False

        # Block localhost
        if parsed.hostname in ('localhost', '127.0.0.1', '::1'):
            return False

        # Resolve hostname and check IP
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))

            # Block private IPs
            if ip.is_private or ip.is_loopback:
                return False
        except (socket.gaierror, ValueError):
            return False

        return True

    except Exception:
        return False


def sanitize_source_uri(source_uri: str) -> str:
    """
    Sanitize source_uri to filename only.

    Prevents leaking directory structure via webhooks.
    """
    return Path(source_uri).name


def send_webhook(url: str, payload: dict, secret: str | None = None, retry_count: int = 2, retry_delay_sec: int = 5) -> bool:
    """
    Send webhook notification with retry logic.

    Returns True if successful, False otherwise.
    """
    import requests

    # Validate URL
    if not validate_webhook_url(url):
        print(f"Invalid webhook URL: {url}")
        return False

    # Add signature if secret provided
    headers = {"Content-Type": "application/json"}
    if secret:
        from .webhook_signature import generate_signature
        signature = generate_signature(payload, secret)
        headers["X-Chinvex-Signature"] = signature

    # Retry loop
    for attempt in range(retry_count + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code < 300:
                return True
            else:
                print(f"Webhook failed with status {response.status_code}")

        except requests.RequestException as e:
            print(f"Webhook request failed: {e}")

        # Retry delay
        if attempt < retry_count:
            time.sleep(retry_delay_sec)

    return False


def create_watch_hit_payload(watch_id: str, query: str, hits: list[dict]) -> dict:
    """
    Create webhook payload for watch hit event.

    Includes snippet (first 200 chars) only, not full chunk text.
    Sanitizes source_uri to filename only.
    """
    formatted_hits = []
    for hit in hits:
        formatted_hits.append({
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "snippet": hit.get("text", "")[:200],
            "source": sanitize_source_uri(hit.get("source_uri", "unknown"))
        })

    return {
        "event": "watch_hit",
        "watch_id": watch_id,
        "query": query,
        "hits": formatted_hits
    }
```

**Step 4: Add requests dependency check**

Note: `requests` library may need to be added to requirements. Check if it's already present.

**Step 5: Run tests**

Run: `python -m pytest tests/test_webhook_notifications.py -v`
Expected: Tests pass

**Step 6: Commit**

```bash
git add src/chinvex/notifications.py tests/test_webhook_notifications.py
git commit -m "feat(notifications): add webhook implementation

- Validate HTTPS and block private IPs
- Sanitize source_uri to filename only
- Implement retry logic with configurable delay
- Create watch hit payload with snippet (200 chars)
- Add security validations per spec

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 23: Webhook signature generation

**Files:**
- Create: `src/chinvex/webhook_signature.py`
- Create: `tests/test_webhook_signature.py`

**Step 1: Write failing test**

Create `tests/test_webhook_signature.py`:

```python
"""Test webhook signature generation."""
import pytest
import hmac
import hashlib
import json


def test_generate_signature_creates_hmac():
    """Test that signature is HMAC-SHA256."""
    from chinvex.webhook_signature import generate_signature

    payload = {"event": "test"}
    secret = "test_secret"

    signature = generate_signature(payload, secret)

    # Should be sha256=<hex>
    assert signature.startswith("sha256=")
    assert len(signature) == 71  # "sha256=" + 64 hex chars


def test_verify_signature_validates_correctly():
    """Test that signature verification works."""
    from chinvex.webhook_signature import generate_signature, verify_signature

    payload = {"event": "test", "data": "value"}
    secret = "test_secret"

    signature = generate_signature(payload, secret)

    assert verify_signature(payload, signature, secret) is True
    assert verify_signature(payload, "sha256=invalid", secret) is False
    assert verify_signature(payload, signature, "wrong_secret") is False


def test_signature_is_deterministic():
    """Test that same payload + secret = same signature."""
    from chinvex.webhook_signature import generate_signature

    payload = {"event": "test"}
    secret = "test_secret"

    sig1 = generate_signature(payload, secret)
    sig2 = generate_signature(payload, secret)

    assert sig1 == sig2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webhook_signature.py::test_generate_signature_creates_hmac -v`
Expected: FAIL with "module not found"

**Step 3: Implement signature module**

Create `src/chinvex/webhook_signature.py`:

```python
"""Webhook signature generation and verification."""
import hmac
import hashlib
import json


def generate_signature(payload: dict, secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.

    Returns signature in format: "sha256=<hex>"
    """
    # Serialize payload to canonical JSON
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')

    # Generate HMAC
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    return f"sha256={signature}"


def verify_signature(payload: dict, signature: str, secret: str) -> bool:
    """
    Verify webhook signature.

    Uses constant-time comparison to prevent timing attacks.
    """
    expected = generate_signature(payload, secret)

    # Constant-time comparison
    return hmac.compare_digest(expected, signature)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_webhook_signature.py -v`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/chinvex/webhook_signature.py tests/test_webhook_signature.py
git commit -m "feat(notifications): add webhook signature generation

- Implement HMAC-SHA256 signature
- Format: sha256=<hex>
- Use constant-time comparison for verification
- Canonical JSON serialization for consistency

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 24: Integrate webhooks with watch runner

**Files:**
- Modify: `src/chinvex/watch/runner.py`
- Modify: `src/chinvex/config.py`
- Create: `tests/test_watch_webhook_integration.py`

**Step 1: Write failing test**

Create `tests/test_watch_webhook_integration.py`:

```python
"""Test webhook integration with watch runner."""
import pytest
from unittest.mock import patch, Mock


@patch('chinvex.notifications.send_webhook')
def test_watch_triggers_webhook(mock_send_webhook):
    """Test that watch hit triggers webhook."""
    from chinvex.watch.runner import trigger_watch_webhook

    mock_send_webhook.return_value = True

    # Mock watch and hits
    watch = Mock(id="test_watch", query="test")
    hits = [
        Mock(chunk_id="abc", score=0.85, text="test text", source_uri="file.txt")
    ]

    # Mock config
    config = Mock()
    config.notifications.enabled = True
    config.notifications.webhook_url = "https://example.com/webhook"
    config.notifications.webhook_secret = "secret"
    config.notifications.min_score_for_notify = 0.75

    # Trigger webhook
    success = trigger_watch_webhook(config, watch, hits)

    assert success is True
    mock_send_webhook.assert_called_once()


def test_watch_webhook_respects_min_score():
    """Test that low-scoring hits don't trigger webhook."""
    from chinvex.watch.runner import should_notify

    # Low score - should not notify
    assert should_notify([Mock(score=0.5)], min_score=0.75) is False

    # High score - should notify
    assert should_notify([Mock(score=0.85)], min_score=0.75) is True


@patch('chinvex.notifications.send_webhook')
def test_watch_webhook_failure_does_not_block(mock_send_webhook):
    """Test that webhook failure doesn't block ingest."""
    from chinvex.watch.runner import trigger_watch_webhook

    mock_send_webhook.return_value = False  # Webhook fails

    # Should return False but not raise exception
    watch = Mock(id="test_watch", query="test")
    hits = [Mock(chunk_id="abc", score=0.85)]
    config = Mock()
    config.notifications.enabled = True

    success = trigger_watch_webhook(config, watch, hits)

    assert success is False
    # No exception raised
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_watch_webhook_integration.py::test_watch_triggers_webhook -v`
Expected: FAIL with "function not defined"

**Step 3: Add NotificationsConfig to config**

Edit `src/chinvex/config.py`, add:

```python
@dataclass
class NotificationsConfig:
    enabled: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""  # Or "env:CHINVEX_WEBHOOK_SECRET"
    notify_on: list[str] = field(default_factory=lambda: ["watch_hit"])
    min_score_for_notify: float = 0.75
    retry_count: int = 2
    retry_delay_sec: int = 5


@dataclass
class AppConfig:
    # ... existing fields ...
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
```

**Step 4: Implement webhook integration in watch runner**

Edit `src/chinvex/watch/runner.py`, add:

```python
def should_notify(hits: list, min_score: float) -> bool:
    """Check if any hit meets minimum score threshold."""
    return any(hit.score >= min_score for hit in hits)


def trigger_watch_webhook(config, watch, hits: list) -> bool:
    """
    Trigger webhook notification for watch hit.

    Returns True if webhook sent successfully, False otherwise.
    Does NOT raise exceptions (failures are logged).
    """
    from ..notifications import send_webhook, create_watch_hit_payload

    if not config.notifications.enabled:
        return False

    if not config.notifications.webhook_url:
        return False

    # Check min score threshold
    if not should_notify(hits, config.notifications.min_score_for_notify):
        return False

    # Create payload
    hits_data = [
        {
            "chunk_id": h.chunk_id,
            "score": h.score,
            "text": h.text if hasattr(h, 'text') else h.snippet,
            "source_uri": h.source_uri if hasattr(h, 'source_uri') else "unknown"
        }
        for h in hits
    ]

    payload = create_watch_hit_payload(watch.id, watch.query, hits_data)

    # Resolve secret
    secret = config.notifications.webhook_secret
    if secret.startswith("env:"):
        import os
        env_var = secret[4:]
        secret = os.environ.get(env_var, "")

    # Send webhook (with retry)
    try:
        return send_webhook(
            config.notifications.webhook_url,
            payload,
            secret=secret if secret else None,
            retry_count=config.notifications.retry_count,
            retry_delay_sec=config.notifications.retry_delay_sec
        )
    except Exception as e:
        print(f"Webhook notification failed: {e}")
        return False
```

Then integrate into the watch execution logic:

```python
# In run_watches() or similar function:
def run_watches(config, run_id: str):
    """Run all watches and log hits."""
    for watch in config.watches:
        hits = search_hybrid(config, watch.query, k=20)

        if hits:
            # Log to history (existing)
            log_watch_hits(config, watch, hits, run_id)

            # NEW: Trigger webhook (don't block on failure)
            trigger_watch_webhook(config, watch, hits)
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_watch_webhook_integration.py -v`
Expected: Tests pass

**Step 6: Manual test**

Configure webhook in config, trigger watch:
```bash
chinvex ingest --context Test
```
Expected: Webhook POST sent if watch triggers

**Step 7: Commit**

```bash
git add src/chinvex/watch/runner.py src/chinvex/config.py tests/test_watch_webhook_integration.py
git commit -m "feat(watch): integrate webhooks with watch runner

- Trigger webhook when watch hits occur
- Respect min_score_for_notify threshold
- Resolve env:VAR_NAME secrets from environment
- Continue on webhook failure (don't block ingest)
- Add NotificationsConfig to AppConfig

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: Gateway Extras (P3.5)

---

### Task 25: Redis-backed rate limiting

**Files:**
- Create: `src/chinvex/gateway/rate_limit_redis.py`
- Modify: `src/chinvex/gateway/app.py`
- Create: `tests/test_redis_rate_limiting.py`

**Step 1: Write failing test**

Create `tests/test_redis_rate_limiting.py`:

```python
"""Test Redis-backed rate limiting."""
import pytest
from unittest.mock import Mock, patch


def test_redis_rate_limiter_allows_within_limit():
    """Test that requests within limit are allowed."""
    from chinvex.gateway.rate_limit_redis import RedisRateLimiter

    # Mock Redis client
    mock_redis = Mock()
    mock_redis.get.return_value = b"5"  # 5 requests

    limiter = RedisRateLimiter(mock_redis, requests_per_minute=60)

    assert limiter.check_rate_limit("client_id") is True


def test_redis_rate_limiter_blocks_over_limit():
    """Test that requests over limit are blocked."""
    from chinvex.gateway.rate_limit_redis import RedisRateLimiter

    mock_redis = Mock()
    mock_redis.get.return_value = b"65"  # Over 60

    limiter = RedisRateLimiter(mock_redis, requests_per_minute=60)

    assert limiter.check_rate_limit("client_id") is False


def test_redis_fallback_to_memory_on_failure():
    """Test fallback to in-memory limiter if Redis unavailable."""
    from chinvex.gateway.rate_limit_redis import create_rate_limiter

    # Redis connection fails
    with patch('redis.Redis', side_effect=Exception("Connection failed")):
        limiter = create_rate_limiter("redis://localhost:6379")

    # Should fallback to memory limiter
    assert limiter is not None
    # TODO: Verify it's using memory backend
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_redis_rate_limiting.py::test_redis_rate_limiter_allows_within_limit -v`
Expected: FAIL with "module not found"

**Step 3: Implement Redis rate limiter**

Create `src/chinvex/gateway/rate_limit_redis.py`:

```python
"""Redis-backed rate limiting."""
import time
from typing import Optional


class RedisRateLimiter:
    """Redis-backed rate limiter using sliding window."""

    def __init__(self, redis_client, requests_per_minute: int = 60, requests_per_hour: int = 500):
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limit.

        Returns True if allowed, False if rate limited.
        """
        now = int(time.time())

        # Check minute window
        minute_key = f"ratelimit:{client_id}:minute:{now // 60}"
        try:
            count = self.redis.incr(minute_key)
            if count == 1:
                self.redis.expire(minute_key, 120)  # 2 minutes TTL

            if count > self.requests_per_minute:
                return False
        except Exception as e:
            print(f"Redis rate limit check failed: {e}")
            return True  # Fail open

        # Check hour window
        hour_key = f"ratelimit:{client_id}:hour:{now // 3600}"
        try:
            count = self.redis.incr(hour_key)
            if count == 1:
                self.redis.expire(hour_key, 7200)  # 2 hours TTL

            if count > self.requests_per_hour:
                return False
        except Exception:
            pass

        return True


class MemoryRateLimiter:
    """In-memory rate limiter fallback."""

    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.counts = {}  # {client_id: [(timestamp, count), ...]}

    def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit using in-memory counters."""
        now = time.time()

        # Clean old entries
        if client_id in self.counts:
            self.counts[client_id] = [
                (ts, cnt) for ts, cnt in self.counts[client_id]
                if now - ts < 3600  # Keep last hour
            ]
        else:
            self.counts[client_id] = []

        # Count requests in windows
        minute_count = sum(cnt for ts, cnt in self.counts[client_id] if now - ts < 60)
        hour_count = sum(cnt for ts, cnt in self.counts[client_id] if now - ts < 3600)

        if minute_count >= self.requests_per_minute or hour_count >= self.requests_per_hour:
            return False

        # Record request
        self.counts[client_id].append((now, 1))
        return True


def create_rate_limiter(redis_url: Optional[str], requests_per_minute: int = 60, requests_per_hour: int = 500):
    """
    Create rate limiter with Redis or memory backend.

    Falls back to memory if Redis unavailable.
    """
    if redis_url:
        try:
            import redis
            client = redis.from_url(redis_url)
            client.ping()  # Test connection
            print(f"Using Redis rate limiter: {redis_url}")
            return RedisRateLimiter(client, requests_per_minute, requests_per_hour)
        except Exception as e:
            print(f"Redis connection failed, using memory fallback: {e}")

    return MemoryRateLimiter(requests_per_minute, requests_per_hour)
```

**Step 4: Integrate with gateway**

Edit `src/chinvex/gateway/app.py` (or wherever gateway middleware is):

```python
# At gateway initialization:
from .rate_limit_redis import create_rate_limiter

rate_limiter = create_rate_limiter(
    redis_url=config.gateway.rate_limit.redis_url,
    requests_per_minute=config.gateway.rate_limit.requests_per_minute,
    requests_per_hour=config.gateway.rate_limit.requests_per_hour
)

# In request handler:
def rate_limit_middleware(client_id: str):
    if not rate_limiter.check_rate_limit(client_id):
        return {"error": "Rate limit exceeded"}, 429
```

**Step 5: Add rate limit config**

Edit `src/chinvex/config.py`:

```python
@dataclass
class RateLimitConfig:
    backend: str = "memory"  # "memory" or "redis"
    redis_url: str | None = None
    requests_per_minute: int = 60
    requests_per_hour: int = 500


@dataclass
class GatewayConfig:
    # ... existing fields ...
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_redis_rate_limiting.py -v`
Expected: Tests pass

**Step 7: Commit**

```bash
git add src/chinvex/gateway/rate_limit_redis.py src/chinvex/gateway/app.py src/chinvex/config.py tests/test_redis_rate_limiting.py
git commit -m "feat(gateway): add Redis-backed rate limiting

- Implement sliding window rate limiter with Redis
- Fallback to in-memory limiter if Redis unavailable
- Support per-minute and per-hour limits
- Add RateLimitConfig to GatewayConfig
- Log warning on fallback

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 26: Prometheus metrics endpoint

**Files:**
- Create: `src/chinvex/gateway/metrics.py`
- Modify: `src/chinvex/gateway/app.py`
- Create: `tests/test_prometheus_metrics.py`

**Step 1: Write failing test**

Create `tests/test_prometheus_metrics.py`:

```python
"""Test Prometheus metrics endpoint."""
import pytest
from unittest.mock import Mock


def test_metrics_endpoint_requires_auth():
    """Test that /metrics requires bearer token."""
    from chinvex.gateway.app import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # No auth - should fail
    response = client.get("/metrics")
    assert response.status_code == 401

    # With auth - should succeed
    response = client.get("/metrics", headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200


def test_metrics_endpoint_returns_prometheus_format():
    """Test that metrics are in Prometheus format."""
    from chinvex.gateway.metrics import generate_metrics

    metrics = generate_metrics()

    # Should contain metric lines
    assert "chinvex_requests_total" in metrics
    assert "chinvex_request_duration_seconds" in metrics
    assert "chinvex_grounded_ratio" in metrics


def test_metrics_track_request_counts():
    """Test that request counter increments."""
    from chinvex.gateway.metrics import MetricsCollector

    collector = MetricsCollector()

    collector.record_request(endpoint="/search", status_code=200)
    collector.record_request(endpoint="/search", status_code=200)
    collector.record_request(endpoint="/evidence", status_code=404)

    metrics = collector.get_metrics()

    # Should have counts per endpoint/status
    assert metrics["requests"]["/search"][200] == 2
    assert metrics["requests"]["/evidence"][404] == 1


def test_metrics_track_grounded_ratio():
    """Test that grounded ratio is tracked."""
    from chinvex.gateway.metrics import MetricsCollector

    collector = MetricsCollector()

    collector.record_grounded_response(True)
    collector.record_grounded_response(True)
    collector.record_grounded_response(False)

    metrics = collector.get_metrics()

    assert metrics["grounded"]["total"] == 3
    assert metrics["grounded"]["grounded"] == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prometheus_metrics.py::test_metrics_endpoint_requires_auth -v`
Expected: FAIL with "module not found"

**Step 3: Implement metrics module**

Create `src/chinvex/gateway/metrics.py`:

```python
"""Prometheus metrics collection."""
import time
from collections import defaultdict
from typing import Dict


class MetricsCollector:
    """In-memory metrics collector for Prometheus."""

    def __init__(self):
        self.requests = defaultdict(lambda: defaultdict(int))  # {endpoint: {status: count}}
        self.latencies = defaultdict(list)  # {endpoint: [duration, ...]}
        self.grounded_total = 0
        self.grounded_count = 0

    def record_request(self, endpoint: str, status_code: int, duration: float = 0.0):
        """Record request metrics."""
        self.requests[endpoint][status_code] += 1
        if duration > 0:
            self.latencies[endpoint].append(duration)

    def record_grounded_response(self, grounded: bool):
        """Record grounded response metric."""
        self.grounded_total += 1
        if grounded:
            self.grounded_count += 1

    def get_metrics(self) -> dict:
        """Get current metrics as dict."""
        return {
            "requests": dict(self.requests),
            "latencies": dict(self.latencies),
            "grounded": {
                "total": self.grounded_total,
                "grounded": self.grounded_count
            }
        }


def generate_metrics(collector: MetricsCollector) -> str:
    """
    Generate Prometheus metrics format.

    Returns plain text in Prometheus exposition format.
    """
    lines = []

    # Request counts
    lines.append("# HELP chinvex_requests_total Total requests by endpoint and status")
    lines.append("# TYPE chinvex_requests_total counter")
    for endpoint, statuses in collector.requests.items():
        for status, count in statuses.items():
            lines.append(f'chinvex_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}')

    # Latency histograms (simplified - just avg for now)
    lines.append("# HELP chinvex_request_duration_seconds Request duration")
    lines.append("# TYPE chinvex_request_duration_seconds histogram")
    for endpoint, durations in collector.latencies.items():
        if durations:
            avg = sum(durations) / len(durations)
            lines.append(f'chinvex_request_duration_seconds{{endpoint="{endpoint}"}} {avg:.4f}')

    # Grounded ratio
    lines.append("# HELP chinvex_grounded_ratio Ratio of grounded responses")
    lines.append("# TYPE chinvex_grounded_ratio gauge")
    if collector.grounded_total > 0:
        ratio = collector.grounded_count / collector.grounded_total
        lines.append(f"chinvex_grounded_ratio {ratio:.4f}")
    else:
        lines.append("chinvex_grounded_ratio 0.0")

    return "\n".join(lines) + "\n"


# Global metrics collector (ephemeral - resets on restart)
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics_collector
```

**Step 4: Add metrics endpoint to gateway**

Edit `src/chinvex/gateway/app.py`:

```python
from .metrics import get_metrics_collector, generate_metrics

# At app initialization:
metrics_collector = get_metrics_collector()

# Add metrics endpoint
@app.get("/metrics")
async def metrics_endpoint(authorization: str = Header(None)):
    """
    Prometheus metrics endpoint.

    Requires bearer token authentication.
    """
    # Check auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization[7:]
    if not verify_token(token):  # Use existing token verification
        raise HTTPException(status_code=401, detail="Invalid token")

    # Generate metrics
    metrics_text = generate_metrics(metrics_collector)

    return Response(content=metrics_text, media_type="text/plain")


# Add middleware to track requests
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    metrics_collector.record_request(
        endpoint=request.url.path,
        status_code=response.status_code,
        duration=duration
    )

    return response
```

**Step 5: Add metrics config**

Edit `src/chinvex/config.py`:

```python
@dataclass
class GatewayConfig:
    # ... existing fields ...
    metrics_enabled: bool = True
    metrics_auth_required: bool = True
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_prometheus_metrics.py -v`
Expected: Tests pass

**Step 7: Manual test**

Start gateway and check metrics:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/metrics
```
Expected: Prometheus format metrics displayed

**Step 8: Commit**

```bash
git add src/chinvex/gateway/metrics.py src/chinvex/gateway/app.py src/chinvex/config.py tests/test_prometheus_metrics.py
git commit -m "feat(gateway): add Prometheus metrics endpoint

- Add /metrics endpoint with bearer auth
- Track request counts per endpoint/status
- Track request latency histograms
- Track grounded response ratio
- Ephemeral metrics (reset on restart)
- Add metrics_enabled and metrics_auth_required config

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Testing Strategy

### Unit Tests
Run all tests after completing all tasks:
```bash
python -m pytest tests/ -v
```

Expected: All tests pass

### Integration Tests
Test CLI commands:
```bash
# Archive tier
chinvex archive run --context Test --older-than 180d
chinvex archive list --context Test
chinvex search --context Test "query" --include-archive

# Webhooks (with test webhook receiver)
# Configure webhook_url in config
chinvex ingest --context Test

# Gateway metrics
curl -H "Authorization: Bearer <token>" http://localhost:8000/metrics
```

### Manual Acceptance Tests
Follow acceptance tests from P3 spec Section 13.

---

## Deployment Notes

1. **Archive migration** runs automatically on first init_storage() after upgrade
2. **Webhook configuration** is opt-in (enabled: false by default)
3. **Redis rate limiting** is optional (falls back to memory)
4. **Prometheus metrics** are ephemeral (reset on restart)

---

## Configuration Summary

Final P3 config structure (from spec):

```json
{
  "archive": {
    "enabled": true,
    "age_threshold_days": 180,
    "auto_archive_on_ingest": true,
    "archive_penalty": 0.8
  },
  "notifications": {
    "enabled": false,
    "webhook_url": "",
    "webhook_secret": "env:CHINVEX_WEBHOOK_SECRET",
    "notify_on": ["watch_hit"],
    "min_score_for_notify": 0.75,
    "retry_count": 2,
    "retry_delay_sec": 5
  },
  "gateway": {
    "rate_limit": {
      "backend": "memory",
      "redis_url": null,
      "requests_per_minute": 60,
      "requests_per_hour": 500
    },
    "metrics_enabled": true,
    "metrics_auth_required": true
  }
}
```

---

## Completion Checklist

After implementing all tasks:

- [ ] All unit tests pass
- [ ] Archive commands work (run, list, restore, purge)
- [ ] Search respects archive filtering
- [ ] Auto-archive runs on ingest
- [ ] Webhooks fire on watch hits
- [ ] Webhook signatures validate
- [ ] Redis rate limiting works (or falls back)
- [ ] Prometheus metrics endpoint returns data
- [ ] Manual acceptance tests pass
- [ ] Documentation updated
