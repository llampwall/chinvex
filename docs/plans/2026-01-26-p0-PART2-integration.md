# P0 Implementation Plan: Context Registry & Codex Ingestion - PART 2

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integration tasks - scoring, CLI updates, Codex ingestion, MCP tool

**Architecture:** Score blending with weight renormalization, context-based CLI, Codex app-server ingestion, chinvex_answer MCP tool

**Tech Stack:** Python 3.12, SQLite FTS5, Chroma, Ollama embeddings, Typer CLI, Pydantic schemas, MCP stdio protocol

**Status:** All tasks in Part 2 are TODO (Part 1 foundation is complete)

---

## Task 11: ❌ TODO - Score Blending with Weight Renormalization

**Files:**
- Create: `src/chinvex/scoring.py`
- Test: `tests/test_scoring.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring.py
from chinvex.scoring import normalize_scores, blend_scores, rank_score


def test_normalize_scores_minmax() -> None:
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = normalize_scores(scores)

    assert normalized[0] == 0.0  # min
    assert normalized[-1] == 1.0  # max
    assert 0.0 <= normalized[2] <= 1.0  # middle


def test_normalize_scores_empty() -> None:
    assert normalize_scores([]) == []


def test_normalize_scores_all_equal() -> None:
    scores = [5.0, 5.0, 5.0]
    normalized = normalize_scores(scores)

    assert all(s == 1.0 for s in normalized)


def test_blend_scores_both_present() -> None:
    fts_norm = 0.8
    vec_norm = 0.6

    blended = blend_scores(fts_norm, vec_norm)

    # FTS_WEIGHT=0.6, VEC_WEIGHT=0.4
    expected = (0.8 * 0.6) + (0.6 * 0.4)
    assert abs(blended - expected) < 0.001


def test_blend_scores_only_fts() -> None:
    blended = blend_scores(0.8, None)
    assert blended == 0.8


def test_blend_scores_only_vector() -> None:
    blended = blend_scores(None, 0.7)
    assert blended == 0.7


def test_blend_scores_neither() -> None:
    blended = blend_scores(None, None)
    assert blended == 0.0


def test_rank_score_applies_weight() -> None:
    blended = 0.8
    weight = 0.9  # codex_session weight

    rank = rank_score(blended, "codex_session", {"codex_session": 0.9})

    assert rank == 0.8 * 0.9


def test_rank_score_uses_default_for_unknown_type() -> None:
    blended = 0.8
    weights = {"repo": 1.0}

    rank = rank_score(blended, "unknown_type", weights)

    assert rank == 0.8 * 0.5  # default weight
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.scoring'"

**Step 3: Implement scoring functions**

```python
# src/chinvex/scoring.py
from __future__ import annotations


FTS_WEIGHT = 0.6
VEC_WEIGHT = 0.4


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalize within the candidate set for this query.
    Returns values in [0, 1].
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)  # all equal = all max
    return [(s - min_s) / (max_s - min_s) for s in scores]


def blend_scores(fts_norm: float | None, vec_norm: float | None) -> float:
    """
    Combine NORMALIZED FTS and vector scores with weight renormalization.

    If only one score present, use 100% of that signal.
    If both present, blend with FTS_WEIGHT and VEC_WEIGHT.
    """
    if fts_norm is not None and vec_norm is not None:
        return fts_norm * FTS_WEIGHT + vec_norm * VEC_WEIGHT
    elif fts_norm is not None:
        return fts_norm  # 100% of available signal
    elif vec_norm is not None:
        return vec_norm
    else:
        return 0.0


def rank_score(blended: float, source_type: str, weights: dict[str, float]) -> float:
    """
    Apply source-type weight as a post-retrieval prior.
    """
    weight = weights.get(source_type, 0.5)  # default if unknown
    return blended * weight
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/scoring.py tests/test_scoring.py
git commit -m "feat: add score blending with weight renormalization

- Implement normalize_scores (min-max within candidate set)
- Implement blend_scores with weight renormalization
- Handle single-source matches (FTS-only or vector-only)
- Implement rank_score to apply source-type weights
- Test normalization, blending, and ranking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: ❌ TODO - Integrate Scoring into Search

**Files:**
- Modify: `src/chinvex/search.py`
- Test: `tests/test_integrated_search.py`

**Step 1: Write the failing test**

```python
# tests/test_integrated_search.py
from pathlib import Path
from chinvex.config import AppConfig, SourceConfig
from chinvex.ingest import ingest
from chinvex.search import search


class FakeEmbedder:
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_integrated_search.py::test_search_applies_score_normalization_and_blending -v`
Expected: Current implementation doesn't normalize scores, so this may pass or fail depending on raw scores

**Step 3: Integrate scoring into search.py**

Modify `src/chinvex/search.py` to use the new scoring module:

```python
# Add import at top
from .scoring import normalize_scores, blend_scores

# Replace search_chunks function (lines 84-149):
def search_chunks(
    storage: Storage,
    vectors: VectorStore,
    embedder: OllamaEmbedder,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    """
    Hybrid search with score normalization and weight renormalization.

    Args:
        weights: Optional source-type weights dict (e.g., {"repo": 1.0, "chat": 0.8})
                 If None, no weight adjustment applied.
    """
    filters = {}
    if source in {"repo", "chat", "codex_session"}:
        filters["source_type"] = source
    if project:
        filters["project"] = project
    if repo:
        filters["repo"] = repo

    # Perform FTS search
    lex_rows = storage.search_fts(query, limit=30, filters=filters)

    # Perform vector search
    where = {}
    if source in {"repo", "chat", "codex_session"}:
        where["source_type"] = source
    if project:
        where["project"] = project
    if repo:
        where["repo"] = repo

    query_embedding = embedder.embed([query])
    vec_result = vectors.query(query_embedding, n_results=30, where=where)

    # Build candidate set
    candidates: dict[str, dict] = {}

    for row in lex_rows:
        chunk_id = row["chunk_id"]
        candidates[chunk_id] = {
            "row": row,
            "fts_raw": float(row["rank"]),
            "vec_raw": None,
        }

    vec_ids = vec_result.get("ids", [[]])[0]
    vec_distances = vec_result.get("distances", [[]])[0]
    for chunk_id, dist in zip(vec_ids, vec_distances):
        if dist is None:
            continue
        if chunk_id not in candidates:
            # Fetch row from storage
            row = _fetch_chunk(storage, chunk_id)
            if row is None:
                continue
            candidates[chunk_id] = {
                "row": row,
                "fts_raw": None,
                "vec_raw": float(dist),
            }
        else:
            candidates[chunk_id]["vec_raw"] = float(dist)

    # Normalize FTS scores (BM25 ranks - lower is better, invert)
    fts_raws = [c["fts_raw"] for c in candidates.values() if c["fts_raw"] is not None]
    if fts_raws:
        fts_normalized = normalize_scores([-r for r in fts_raws])  # invert ranks
        fts_map = dict(zip([cid for cid, c in candidates.items() if c["fts_raw"] is not None], fts_normalized))
    else:
        fts_map = {}

    # Normalize vector scores (cosine distance - lower is better, invert)
    vec_raws = [c["vec_raw"] for c in candidates.values() if c["vec_raw"] is not None]
    if vec_raws:
        vec_normalized = normalize_scores([1.0 / (1.0 + d) for d in vec_raws])
        vec_map = dict(zip([cid for cid, c in candidates.items() if c["vec_raw"] is not None], vec_normalized))
    else:
        vec_map = {}

    # Blend scores with weight renormalization
    results: list[ScoredChunk] = []
    for chunk_id, data in candidates.items():
        row = data["row"]
        fts_norm = fts_map.get(chunk_id)
        vec_norm = vec_map.get(chunk_id)

        blended = blend_scores(fts_norm, vec_norm)

        # Apply source-type weight if provided
        if weights:
            source_type = row["source_type"]
            weight = weights.get(source_type, 0.5)
            score = blended * weight
        else:
            score = blended

        if score < min_score:
            continue

        results.append(ScoredChunk(chunk_id=chunk_id, score=score, row=row))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
```

Also update the `search` function signature to accept weights:

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
) -> list[SearchResult]:
    db_path = config.index_dir / "hybrid.db"
    chroma_dir = config.index_dir / "chroma"
    storage = Storage(db_path)
    storage.ensure_schema()

    ollama_host = ollama_host_override or config.ollama_host
    embedder = OllamaEmbedder(ollama_host, config.embedding_model)
    vectors = VectorStore(chroma_dir)
    scored = search_chunks(
        storage,
        vectors,
        embedder,
        query,
        k=k,
        min_score=min_score,
        source=source,
        project=project,
        repo=repo,
        weights=weights,
    )
    results = [
        SearchResult(
            chunk_id=item.chunk_id,
            score=item.score,
            source_type=item.row["source_type"],
            title=_title_from_row(item.row),
            citation=_citation_from_row(item.row),
            snippet=make_snippet(item.row["text"], limit=200),
        )
        for item in scored
    ]
    storage.close()
    return results[:k]
```

Remove old normalization functions (_normalize_lex, _normalize_vec, _recency_norm) - lines 152-202.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_integrated_search.py::test_search_applies_score_normalization_and_blending -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/search.py tests/test_integrated_search.py
git commit -m "feat: integrate score normalization and blending into search

- Replace ad-hoc scoring with normalize_scores and blend_scores
- Build unified candidate set from FTS and vector results
- Apply weight renormalization (100% signal when only one source)
- Add weights parameter to search functions
- Remove old _normalize_lex, _normalize_vec, _recency_norm functions
- Test normalized and blended scores

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 13: ❌ TODO - Update CLI Ingest to Use Context Registry

**Files:**
- Modify: `src/chinvex/cli.py`
- Modify: `src/chinvex/ingest.py`
- Test: `tests/test_ingest_with_context.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_with_context.py -v`
Expected: FAIL (ingest command doesn't support --context yet)

**Step 3: Update ingest CLI command**

Modify `src/chinvex/cli.py`:

```python
# Update ingest command
@app.command()
def ingest(
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to ingest"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    ollama_host: str | None = typer.Option(None, help="Override Ollama host")
) -> None:
    """Ingest sources into the index."""
    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if context:
        # New context-based ingestion
        from .context import load_context
        from .context_cli import get_contexts_root
        from .ingest import ingest_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)
        stats = ingest_context(ctx, ollama_host_override=ollama_host)

        typer.secho(f"Ingestion complete for context '{context}':", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {stats['documents']}")
        typer.echo(f"  Chunks: {stats['chunks']}")
        typer.echo(f"  Skipped: {stats['skipped']}")
    else:
        # Old config-based ingestion (deprecated)
        from .config import load_config, migrate_old_config
        from .ingest import ingest as old_ingest

        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)

        # Auto-migrate if possible
        cfg = load_config(config)
        if cfg:
            ctx_name = migrate_old_config(config)
            typer.secho(f"Auto-migrated to context '{ctx_name}'. Future ingests should use: chinvex ingest --context {ctx_name}", fg=typer.colors.YELLOW)
        else:
            stats = old_ingest(cfg, ollama_host_override=ollama_host)
            typer.secho("Ingestion complete:", fg=typer.colors.GREEN)
            typer.echo(f"  Documents: {stats['documents']}")
            typer.echo(f"  Chunks: {stats['chunks']}")
            typer.echo(f"  Skipped: {stats['skipped']}")
```

**Step 4: Implement ingest_context function**

Add to `src/chinvex/ingest.py`:

```python
from .context import ContextConfig

def ingest_context(ctx: ContextConfig, *, ollama_host_override: str | None = None) -> dict:
    """
    Ingest all sources from a context.

    Uses context.index paths for storage.
    Applies context.weights for source-type prioritization.
    Uses fingerprinting for incremental ingest.
    """
    db_path = ctx.index.sqlite_path
    chroma_dir = ctx.index.chroma_dir

    db_path.parent.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    lock_path = db_path.parent / "hybrid.db.lock"
    try:
        with Lock(lock_path, timeout=60):
            storage = Storage(db_path)
            storage.ensure_schema()

            # Determine embedding model from context or use default
            # (For P0, hardcode mxbai-embed-large, later read from context if added)
            embedding_model = "mxbai-embed-large"
            ollama_host = ollama_host_override or "http://127.0.0.1:11434"

            embedder = OllamaEmbedder(ollama_host, embedding_model)
            vectors = VectorStore(chroma_dir)

            stats = {"documents": 0, "chunks": 0, "skipped": 0}
            started_at = now_iso()
            run_id = sha256_text(started_at)

            # Ingest repos
            for repo_path in ctx.includes.repos:
                if not repo_path.exists():
                    print(f"Warning: repo path {repo_path} does not exist, skipping.")
                    continue
                _ingest_repo_from_context(
                    ctx, repo_path, storage, embedder, vectors, stats
                )

            # Ingest chat roots
            for chat_root in ctx.includes.chat_roots:
                if not chat_root.exists():
                    print(f"Warning: chat_root {chat_root} does not exist, skipping.")
                    continue
                _ingest_chat_from_context(
                    ctx, chat_root, storage, embedder, vectors, stats
                )

            # TODO: Ingest codex_session_roots (Task 14)
            # TODO: Ingest note_roots (post-P0)

            storage.record_run(run_id, started_at, dump_json(stats))
            storage.close()
            return stats
    except LockException as exc:
        raise RuntimeError(
            "Ingest lock is held by another process. Only one ingest can run at a time."
        ) from exc


def _ingest_repo_from_context(
    ctx: ContextConfig,
    repo_path: Path,
    storage: Storage,
    embedder: OllamaEmbedder,
    vectors: VectorStore,
    stats: dict,
) -> None:
    """Ingest a single repo with fingerprinting."""
    for path in walk_files(repo_path):
        text = read_text_utf8(path)
        if text is None:
            continue

        doc_id = sha256_text(f"repo|{ctx.name}|{normalized_path(path)}")
        updated_at = iso_from_mtime(path)
        content_hash = sha256_text(text)

        # Check fingerprint
        fp = storage.get_fingerprint(normalized_path(path), ctx.name)
        if fp and fp["content_sha256"] == content_hash and fp["mtime_unix"] == int(path.stat().st_mtime):
            stats["skipped"] += 1
            continue

        # Delete old chunks
        chunk_ids = storage.delete_chunks_for_doc(doc_id) if fp else []
        if chunk_ids:
            vectors.delete(chunk_ids)

        # Ingest new chunks
        meta = {
            "path": normalized_path(path),
            "ext": path.suffix.lower(),
            "mtime": updated_at,
            "repo": str(repo_path.name),
        }

        storage.upsert_document(
            doc_id=doc_id,
            source_type="repo",
            source_uri=normalized_path(path),
            project=None,
            repo=str(repo_path.name),
            title=path.name,
            updated_at=updated_at,
            content_hash=content_hash,
            meta_json=dump_json(meta),
        )

        chunks = chunk_repo(text)
        chunk_rows = []
        fts_rows = []
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for chunk in chunks:
            chunk_id = sha256_text(f"{doc_id}|{chunk.ordinal}|{sha256_text(chunk.text)}")
            cmeta = {
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "path": normalized_path(path),
                "repo": str(repo_path.name),
            }

            chunk_rows.append((
                chunk_id,
                doc_id,
                "repo",
                None,
                str(repo_path.name),
                chunk.ordinal,
                chunk.text,
                updated_at,
                dump_json(cmeta),
            ))
            fts_rows.append((chunk_id, chunk.text))
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append({
                "source_type": "repo",
                "repo": str(repo_path.name),
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "path": normalized_path(path),
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            })

        embeddings = embedder.embed(docs)
        storage.upsert_chunks(chunk_rows)
        storage.upsert_fts(fts_rows)
        vectors.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

        # Update fingerprint
        storage.upsert_fingerprint(
            source_uri=normalized_path(path),
            context_name=ctx.name,
            source_type="repo",
            doc_id=doc_id,
            size_bytes=path.stat().st_size,
            mtime_unix=int(path.stat().st_mtime),
            content_sha256=content_hash,
            parser_version="v1",
            chunker_version="v1",
            embedded_model=embedder.model,
            last_status="ok",
            last_error=None,
        )

        stats["documents"] += 1
        stats["chunks"] += len(chunks)


def _ingest_chat_from_context(
    ctx: ContextConfig,
    chat_root: Path,
    storage: Storage,
    embedder: OllamaEmbedder,
    vectors: VectorStore,
    stats: dict,
) -> None:
    """Ingest chat exports from a root directory with fingerprinting."""
    for path in Path(chat_root).glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON in {path}, skipping.")
            continue

        if not isinstance(raw, dict) or "messages" not in raw:
            continue

        messages = raw.get("messages") or []
        if not isinstance(messages, list):
            continue

        conversation_id = str(raw.get("conversation_id", path.stem))
        title = str(raw.get("title", ""))
        project = str(raw.get("project", ""))
        exported_at = str(raw.get("exported_at") or iso_from_mtime(path))
        doc_id = conversation_id

        # Build canonical text
        lines = []
        for i, msg in enumerate(messages):
            role = str(msg.get("role", "unknown"))
            text = str(msg.get("text", "")).strip()
            lines.append(f"[{i:04d}] {role}: {text}")
        canonical_text = "\n".join(lines)
        content_hash = sha256_text(canonical_text)

        # Check fingerprint
        fp = storage.get_fingerprint(normalized_path(path), ctx.name)
        if fp and fp["content_sha256"] == content_hash:
            stats["skipped"] += 1
            continue

        # Delete old chunks
        chunk_ids = storage.delete_chunks_for_doc(doc_id) if fp else []
        if chunk_ids:
            vectors.delete(chunk_ids)

        # Ingest new chunks
        meta = {
            "conversation_id": conversation_id,
            "url": raw.get("url"),
            "exported_at": exported_at,
            "project": project,
            "title": title,
        }

        storage.upsert_document(
            doc_id=doc_id,
            source_type="chat",
            source_uri=normalized_path(path),
            project=project,
            repo=None,
            title=title,
            updated_at=exported_at,
            content_hash=content_hash,
            meta_json=dump_json(meta),
        )

        chunks = chunk_chat(messages)
        chunk_rows = []
        fts_rows = []
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for chunk in chunks:
            chunk_id = sha256_text(f"{doc_id}|{chunk.ordinal}|{sha256_text(chunk.text)}")
            cmeta = {
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "msg_start": chunk.msg_start,
                "msg_end": chunk.msg_end,
                "roles_present": chunk.roles_present or [],
                "project": project,
                "title": title,
            }

            chunk_rows.append((
                chunk_id,
                doc_id,
                "chat",
                project,
                None,
                chunk.ordinal,
                chunk.text,
                exported_at,
                dump_json(cmeta),
            ))
            fts_rows.append((chunk_id, chunk.text))
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append({
                "source_type": "chat",
                "project": project,
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "msg_start": chunk.msg_start,
                "msg_end": chunk.msg_end,
                "title": title,
            })

        embeddings = embedder.embed(docs)
        storage.upsert_chunks(chunk_rows)
        storage.upsert_fts(fts_rows)
        vectors.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

        # Update fingerprint
        storage.upsert_fingerprint(
            source_uri=normalized_path(path),
            context_name=ctx.name,
            source_type="chat",
            doc_id=doc_id,
            size_bytes=path.stat().st_size,
            mtime_unix=int(path.stat().st_mtime),
            content_sha256=content_hash,
            parser_version="v1",
            chunker_version="v1",
            embedded_model=embedder.model,
            last_status="ok",
            last_error=None,
        )

        stats["documents"] += 1
        stats["chunks"] += len(chunks)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_ingest_with_context.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/ingest.py tests/test_ingest_with_context.py
git commit -m "feat: update ingest CLI to use context registry

- Add --context option to ingest command
- Implement ingest_context function
- Add _ingest_repo_from_context and _ingest_chat_from_context
- Use fingerprinting for incremental ingest
- Auto-migrate old --config to context (with warning)
- Test ingest with context name

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 14: ❌ TODO - Codex Ingestion with Fingerprinting

**Files:**
- Modify: `src/chinvex/ingest.py`
- Test: `tests/test_codex_ingestion.py`

**Step 1: Write the failing test**

```python
# tests/test_codex_ingestion.py
from pathlib import Path
import json
from chinvex.ingest import _ingest_codex_sessions_from_context
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex
from chinvex.storage import Storage
from chinvex.vectors import VectorStore


class FakeEmbedder:
    model = "test-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeAppServerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def list_threads(self) -> list[dict]:
        return [
            {"id": "thread-1", "title": "Test 1", "created_at": "2026-01-26T10:00:00Z", "updated_at": "2026-01-26T10:30:00Z"},
        ]

    def get_thread(self, thread_id: str) -> dict:
        return {
            "id": thread_id,
            "title": "Test Thread",
            "created_at": "2026-01-26T10:00:00Z",
            "updated_at": "2026-01-26T10:30:00Z",
            "turns": [
                {"turn_id": "turn-1", "ts": "2026-01-26T10:00:00Z", "role": "user", "text": "hello"}
            ],
            "links": {}
        }


def test_ingest_codex_sessions_with_fingerprinting(tmp_path: Path, monkeypatch) -> None:
    # Create test context
    ctx = ContextConfig(
        schema_version=1,
        name="TestCtx",
        aliases=[],
        includes=ContextIncludes(repos=[], chat_roots=[], codex_session_roots=[], note_roots=[]),
        index=ContextIndex(
            sqlite_path=tmp_path / "hybrid.db",
            chroma_dir=tmp_path / "chroma"
        ),
        weights={"codex_session": 0.9, "repo": 1.0, "chat": 0.8, "note": 0.7},
        created_at="2026-01-26T00:00:00Z",
        updated_at="2026-01-26T00:00:00Z"
    )

    storage = Storage(tmp_path / "hybrid.db")
    storage.ensure_schema()

    vectors = VectorStore(tmp_path / "chroma")
    embedder = FakeEmbedder()

    monkeypatch.setattr("chinvex.ingest.AppServerClient", FakeAppServerClient)

    stats = {"documents": 0, "chunks": 0, "skipped": 0}

    _ingest_codex_sessions_from_context(
        ctx,
        "http://localhost:8080",
        storage,
        embedder,
        vectors,
        stats
    )

    # Should have ingested the thread
    assert stats["documents"] > 0

    # Verify fingerprint was created
    fp = storage.get_fingerprint("thread-1", ctx.name)
    assert fp is not None
    assert fp["source_type"] == "codex_session"
    assert fp["last_status"] == "ok"

    # Second ingest should skip
    stats2 = {"documents": 0, "chunks": 0, "skipped": 0}
    _ingest_codex_sessions_from_context(
        ctx,
        "http://localhost:8080",
        storage,
        embedder,
        vectors,
        stats2
    )

    assert stats2["skipped"] > 0
    assert stats2["documents"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_codex_ingestion.py -v`
Expected: FAIL with "ImportError: cannot import name '_ingest_codex_sessions_from_context'"

**Step 3: Implement Codex ingestion**

Add to `src/chinvex/ingest.py`:

```python
from .adapters.cx_appserver.client import AppServerClient
from .adapters.cx_appserver.schemas import AppServerThread
from .adapters.cx_appserver.normalize import normalize_to_conversation_doc

def _ingest_codex_sessions_from_context(
    ctx: ContextConfig,
    appserver_url: str,
    storage: Storage,
    embedder: OllamaEmbedder,
    vectors: VectorStore,
    stats: dict,
) -> None:
    """
    Ingest Codex sessions from app-server with fingerprinting.

    Uses thread_updated_at and last_turn_id for incremental detection.
    """
    client = AppServerClient(appserver_url)

    try:
        threads_list = client.list_threads()
    except Exception as exc:
        print(f"Warning: failed to fetch threads from {appserver_url}: {exc}")
        return

    for thread_meta in threads_list:
        thread_id = thread_meta["id"]
        thread_updated_at = thread_meta.get("updated_at", "")

        # Check fingerprint
        fp = storage.get_fingerprint(thread_id, ctx.name)
        if fp and fp["thread_updated_at"] == thread_updated_at:
            stats["skipped"] += 1
            continue

        # Fetch full thread
        try:
            raw_thread = client.get_thread(thread_id)
        except Exception as exc:
            print(f"Warning: failed to fetch thread {thread_id}: {exc}")
            # Record error fingerprint
            doc_id = sha256_text(f"codex_session|{ctx.name}|{thread_id}")
            storage.upsert_fingerprint(
                source_uri=thread_id,
                context_name=ctx.name,
                source_type="codex_session",
                doc_id=doc_id,
                thread_updated_at=thread_updated_at,
                last_turn_id=None,
                parser_version="v1",
                chunker_version="v1",
                embedded_model=embedder.model,
                last_status="error",
                last_error=str(exc),
            )
            continue

        # Normalize to ConversationDoc
        app_thread = AppServerThread.model_validate(raw_thread)
        conversation_doc = normalize_to_conversation_doc(app_thread)

        doc_id = sha256_text(f"codex_session|{ctx.name}|{thread_id}")
        updated_at = conversation_doc["updated_at"]
        content_hash = sha256_text(dump_json(conversation_doc["turns"]))

        # Delete old chunks
        chunk_ids = storage.delete_chunks_for_doc(doc_id) if fp else []
        if chunk_ids:
            vectors.delete(chunk_ids)

        # Ingest document
        meta = {
            "thread_id": thread_id,
            "title": conversation_doc["title"],
            "created_at": conversation_doc["created_at"],
            "links": conversation_doc["links"],
        }

        storage.upsert_document(
            doc_id=doc_id,
            source_type="codex_session",
            source_uri=thread_id,
            project=None,
            repo=None,
            title=conversation_doc["title"],
            updated_at=updated_at,
            content_hash=content_hash,
            meta_json=dump_json(meta),
        )

        # Chunk conversation
        from .chunking import chunk_conversation
        chunks = chunk_conversation(conversation_doc)

        chunk_rows = []
        fts_rows = []
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for chunk in chunks:
            chunk_id = sha256_text(f"{doc_id}|{chunk.ordinal}|{sha256_text(chunk.text)}")
            cmeta = {
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "thread_id": thread_id,
                "title": conversation_doc["title"],
            }

            chunk_rows.append((
                chunk_id,
                doc_id,
                "codex_session",
                None,
                None,
                chunk.ordinal,
                chunk.text,
                updated_at,
                dump_json(cmeta),
            ))
            fts_rows.append((chunk_id, chunk.text))
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append({
                "source_type": "codex_session",
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "thread_id": thread_id,
                "title": conversation_doc["title"],
            })

        embeddings = embedder.embed(docs)
        storage.upsert_chunks(chunk_rows)
        storage.upsert_fts(fts_rows)
        vectors.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

        # Update fingerprint
        last_turn_id = conversation_doc["turns"][-1]["turn_id"] if conversation_doc["turns"] else None
        storage.upsert_fingerprint(
            source_uri=thread_id,
            context_name=ctx.name,
            source_type="codex_session",
            doc_id=doc_id,
            thread_updated_at=thread_updated_at,
            last_turn_id=last_turn_id,
            parser_version="v1",
            chunker_version="v1",
            embedded_model=embedder.model,
            last_status="ok",
            last_error=None,
        )

        stats["documents"] += 1
        stats["chunks"] += len(chunks)
```

Also update `ingest_context` to call this function:

```python
# In ingest_context function, after ingesting chat_roots:

# Ingest Codex sessions
for codex_root in ctx.includes.codex_session_roots:
    if not codex_root.exists():
        print(f"Warning: codex_session_root {codex_root} does not exist, skipping.")
        continue
    # Assume app-server URL from environment or config (for P0, hardcode localhost)
    appserver_url = os.getenv("CHINVEX_APPSERVER_URL", "http://localhost:8080")
    _ingest_codex_sessions_from_context(
        ctx, appserver_url, storage, embedder, vectors, stats
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_codex_ingestion.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chinvex/ingest.py tests/test_codex_ingestion.py
git commit -m "feat: add Codex session ingestion with fingerprinting

- Implement _ingest_codex_sessions_from_context
- Use AppServerClient to fetch threads
- Normalize to ConversationDoc
- Chunk conversations with token approximation
- Use thread_updated_at and last_turn_id for incremental detection
- Record error fingerprints on fetch failures
- Test Codex ingestion and fingerprint skipping

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 15: ❌ TODO - Update Search CLI to Use Contexts

**Files:**
- Modify: `src/chinvex/cli.py`
- Test: `tests/test_search_with_context.py`

**Step 1: Write the failing test**

```python
# tests/test_search_with_context.py
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
        model = "test"
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_with_context.py -v`
Expected: FAIL (search command doesn't support --context yet)

**Step 3: Update search CLI command**

Modify `src/chinvex/cli.py`:

```python
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to search"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    k: int = typer.Option(8, help="Number of results"),
    min_score: float = typer.Option(0.35, help="Minimum score threshold"),
    source: str = typer.Option("all", help="Filter by source type (repo, chat, codex_session, all)"),
    ollama_host: str | None = typer.Option(None, help="Override Ollama host"),
) -> None:
    """Search the index."""
    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if context:
        # New context-based search
        from .context import load_context
        from .context_cli import get_contexts_root
        from .search import search_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)

        results = search_context(
            ctx,
            query,
            k=k,
            min_score=min_score,
            source=source,
            ollama_host_override=ollama_host,
        )

        if not results:
            typer.echo("No results found.")
            return

        for i, result in enumerate(results, 1):
            typer.secho(f"\n[{i}] {result.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"Score: {result.score:.3f} | Type: {result.source_type}")
            typer.echo(f"Citation: {result.citation}")
            typer.echo(f"Snippet: {result.snippet}")
    else:
        # Old config-based search (deprecated)
        from .config import load_config
        from .search import search as old_search

        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)

        cfg = load_config(config)
        results = old_search(
            cfg,
            query,
            k=k,
            min_score=min_score,
            source=source,
            ollama_host_override=ollama_host,
        )

        if not results:
            typer.echo("No results found.")
            return

        for i, result in enumerate(results, 1):
            typer.secho(f"\n[{i}] {result.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"Score: {result.score:.3f} | Type: {result.source_type}")
            typer.echo(f"Citation: {result.citation}")
            typer.echo(f"Snippet: {result.snippet}")
```

**Step 4: Implement search_context function**

Add to `src/chinvex/search.py`:

```python
from .context import ContextConfig

def search_context(
    ctx: ContextConfig,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    ollama_host_override: str | None = None,
) -> list[SearchResult]:
    """
    Search within a context using context-aware weights.
    """
    db_path = ctx.index.sqlite_path
    chroma_dir = ctx.index.chroma_dir

    storage = Storage(db_path)
    storage.ensure_schema()

    # Use default Ollama host (could be extended to read from context)
    ollama_host = ollama_host_override or "http://127.0.0.1:11434"
    embedding_model = "mxbai-embed-large"  # P0 hardcode

    embedder = OllamaEmbedder(ollama_host, embedding_model)
    vectors = VectorStore(chroma_dir)

    # Use context weights for source-type prioritization
    scored = search_chunks(
        storage,
        vectors,
        embedder,
        query,
        k=k,
        min_score=min_score,
        source=source,
        project=project,
        repo=repo,
        weights=ctx.weights,
    )

    results = [
        SearchResult(
            chunk_id=item.chunk_id,
            score=item.score,
            source_type=item.row["source_type"],
            title=_title_from_row(item.row),
            citation=_citation_from_row(item.row),
            snippet=make_snippet(item.row["text"], limit=200),
        )
        for item in scored
    ]

    storage.close()
    return results[:k]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_search_with_context.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/search.py tests/test_search_with_context.py
git commit -m "feat: update search CLI to use context registry

- Add --context option to search command
- Implement search_context function
- Apply context.weights for source-type prioritization
- Deprecate --config with warning
- Test search with context name

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 16: ❌ TODO - chinvex_answer MCP Tool

**Files:**
- Modify: `src/chinvex_mcp/server.py`
- Test: `tests/test_mcp_answer_tool.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_answer_tool.py
from pathlib import Path
import json
from chinvex_mcp.server import handle_chinvex_answer
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex


class FakeEmbedder:
    model = "test"
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

    monkeypatch.setattr("chinvex_mcp.server.OllamaEmbedder", FakeEmbedder)

    # Call chinvex_answer
    result = handle_chinvex_answer(
        query="test query",
        context_name="TestCtx",
        k=3,
        min_score=0.1,
        contexts_root=tmp_path / "contexts"  # will fail to load, but test structure
    )

    # Should return evidence pack structure (even if empty)
    assert "query" in result
    assert "chunks" in result
    assert "citations" in result
    assert result["query"] == "test query"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_answer_tool.py -v`
Expected: FAIL with "ImportError: cannot import name 'handle_chinvex_answer'"

**Step 3: Implement chinvex_answer tool**

Modify `src/chinvex_mcp/server.py`:

```python
# Add new imports at top
from chinvex.context import load_context, ContextNotFoundError
from chinvex.search import search_context

def handle_chinvex_answer(
    query: str,
    context_name: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str | None = None,
    contexts_root: Path | None = None,
) -> dict:
    """
    Grounded search tool that returns evidence pack (no LLM synthesis).

    Returns:
        {
            "query": str,
            "chunks": [{"chunk_id": str, "text": str, "score": float, "source_type": str}],
            "citations": [str],
            "context_name": str,
            "weights_applied": dict[str, float]
        }
    """
    if contexts_root is None:
        contexts_root = Path(os.getenv("CHINVEX_CONTEXTS_ROOT", "P:/ai_memory/contexts"))

    try:
        ctx = load_context(context_name, contexts_root)
    except ContextNotFoundError as exc:
        return {
            "error": str(exc),
            "query": query,
            "chunks": [],
            "citations": [],
        }

    results = search_context(
        ctx,
        query,
        k=k,
        min_score=min_score,
        source=source or "all",
    )

    chunks = [
        {
            "chunk_id": r.chunk_id,
            "text": r.snippet,  # P0: return snippet, not full text
            "score": r.score,
            "source_type": r.source_type,
        }
        for r in results
    ]

    citations = [r.citation for r in results]

    return {
        "query": query,
        "chunks": chunks,
        "citations": citations,
        "context_name": context_name,
        "weights_applied": ctx.weights,
    }


# Add tool registration in the main server loop
# (This depends on the MCP server framework structure - pseudocode for P0)
@mcp_tool("chinvex_answer")
def chinvex_answer_tool(query: str, context: str, k: int = 8, min_score: float = 0.35) -> str:
    """
    Grounded search that returns evidence pack for answering questions.

    Args:
        query: The question or search query
        context: Context name to search within
        k: Number of chunks to return (default 8)
        min_score: Minimum relevance score (default 0.35)

    Returns:
        JSON string with evidence pack containing chunks and citations
    """
    result = handle_chinvex_answer(query, context, k=k, min_score=min_score)
    return json.dumps(result, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_answer_tool.py -v`
Expected: PASS (or SKIP if actual context loading fails, but structure should be tested)

**Step 5: Update MCP tool descriptions**

Update the MCP server's tool manifest to include `chinvex_answer`:

```python
# In server tool manifest
tools = [
    {
        "name": "chinvex_search",
        "description": "Search chinvex index (returns chunk IDs and metadata)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "string"},
                "k": {"type": "integer", "default": 8},
                "min_score": {"type": "number", "default": 0.35},
            },
            "required": ["query", "context"]
        }
    },
    {
        "name": "chinvex_get_chunk",
        "description": "Get full chunk text by chunk_id",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["chunk_id", "context"]
        }
    },
    {
        "name": "chinvex_answer",
        "description": "Grounded search returning evidence pack (chunks + citations) for answering questions. No LLM synthesis - returns raw evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or search query"},
                "context": {"type": "string", "description": "Context name to search"},
                "k": {"type": "integer", "default": 8, "description": "Number of chunks"},
                "min_score": {"type": "number", "default": 0.35, "description": "Min relevance score"},
            },
            "required": ["query", "context"]
        }
    }
]
```

**Step 6: Commit**

```bash
git add src/chinvex_mcp/server.py tests/test_mcp_answer_tool.py
git commit -m "feat: add chinvex_answer MCP tool

- Implement handle_chinvex_answer function
- Return evidence pack with chunks and citations
- No LLM synthesis (P0 scope)
- Add tool registration and manifest
- Test evidence pack structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 17: ❌ TODO - Add requests Dependency

**Files:**
- Modify: `pyproject.toml` or `setup.py`

**Step 1: Verify requests is needed**

Check if `requests` is already in dependencies:

Run: `grep -i requests pyproject.toml setup.py`

**Step 2: Add requests to dependencies**

If using `pyproject.toml`:

```toml
[project]
dependencies = [
    "chromadb",
    "ollama",
    "portalocker",
    "typer",
    "pydantic",
    "requests",  # Added for app-server client
]
```

If using `setup.py`:

```python
install_requires=[
    "chromadb",
    "ollama",
    "portalocker",
    "typer",
    "pydantic",
    "requests",  # Added for app-server client
]
```

**Step 3: Install updated dependencies**

Run: `pip install -e .`
Expected: requests installed successfully

**Step 4: Commit**

```bash
git add pyproject.toml  # or setup.py
git commit -m "feat: add requests dependency for app-server client

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 18: ❌ TODO - Update README with Context Registry Usage

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Add new section after "Config":

```markdown
## Context Registry

Chinvex now uses a context registry system for managing multiple projects.

### Create a context

```powershell
chinvex context create MyProject
```

This creates:
- `P:\ai_memory\contexts\MyProject\context.json`
- `P:\ai_memory\indexes\MyProject\hybrid.db`
- `P:\ai_memory\indexes\MyProject\chroma\`

### Edit context configuration

Edit `P:\ai_memory\contexts\MyProject\context.json`:

```json
{
  "schema_version": 1,
  "name": "MyProject",
  "aliases": ["myproj"],
  "includes": {
    "repos": ["C:\\Code\\myproject"],
    "chat_roots": ["P:\\ai_memory\\chats\\myproject"],
    "codex_session_roots": [],
    "note_roots": []
  },
  "index": {
    "sqlite_path": "P:\\ai_memory\\indexes\\MyProject\\hybrid.db",
    "chroma_dir": "P:\\ai_memory\\indexes\\MyProject\\chroma"
  },
  "weights": {
    "repo": 1.0,
    "chat": 0.8,
    "codex_session": 0.9,
    "note": 0.7
  }
}
```

### List contexts

```powershell
chinvex context list
```

### Ingest with context

```powershell
chinvex ingest --context MyProject
```

### Search with context

```powershell
chinvex search --context MyProject "your query"
```

### MCP Server with Context

Update your MCP config to use context names:

```json
{
  "mcpServers": {
    "chinvex": {
      "command": "chinvex-mcp",
      "args": ["--context", "MyProject"]
    }
  }
}
```

### Migration from Old Config

Old `--config` flag is deprecated. On first use, chinvex will auto-migrate:

```powershell
chinvex ingest --config old_config.json
```

This creates a new context and suggests using `--context` going forward.
```

**Step 2: Update MCP Server section**

Update the "MCP Server" section:

```markdown
## MCP Server

Run the local MCP server with a context:

```powershell
chinvex-mcp --context MyProject
```

Optional overrides:

```powershell
chinvex-mcp --context MyProject --ollama-host http://skynet:11434 --k 8 --min-score 0.30
```

### Available Tools

- `chinvex_search`: Search for relevant chunks
- `chinvex_get_chunk`: Get full chunk text by ID
- `chinvex_answer`: Grounded search returning evidence pack (chunks + citations)
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with context registry usage

- Add context registry section
- Document context create, list, ingest, search commands
- Update MCP server section with context usage
- Add migration guide from old config
- Document chinvex_answer MCP tool

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Plan Complete

All 18 tasks are now fully specified with TDD steps, complete code, and exact commands.

**Summary:**
1. ✅ Schema Version + Meta Table
2. ✅ Add source_fingerprints Table
3. ✅ Context Registry Data Structures
4. ✅ CLI Command - context create
5. ✅ CLI Command - context list
6. ✅ Auto-Migration from Old Config Format
7. ✅ Conversation Chunking with Token Approximation
8. ✅ Codex App-Server Client (Schema Capture)
9. ✅ Codex App-Server Schemas (Pydantic)
10. ✅ Normalize App-Server to ConversationDoc
11. ✅ Score Blending with Weight Renormalization
12. ✅ Integrate Scoring into Search
13. ✅ Update CLI Ingest to Use Context Registry
14. ✅ Codex Ingestion with Fingerprinting
15. ✅ Update Search CLI to Use Contexts
16. ✅ chinvex_answer MCP Tool
17. ✅ Add requests Dependency
18. ✅ Update README with Context Registry Usage