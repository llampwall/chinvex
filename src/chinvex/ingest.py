from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

from portalocker import Lock, LockException

from .adapters.cx_appserver.client import AppServerClient
from .adapters.cx_appserver.normalize import normalize_to_conversation_doc
from .adapters.cx_appserver.schemas import AppServerThread
from .chunking import chunk_chat, chunk_conversation, chunk_key, chunk_repo
from .config import AppConfig, SourceConfig
from .context import ContextConfig
from .embed import OllamaEmbedder
from .hooks import post_ingest_hook
from .storage import Storage
from .util import dump_json, iso_from_mtime, normalized_path, now_iso, read_text_utf8, sha256_text, walk_files
from .vectors import VectorStore


@dataclass
class IngestRunResult:
    """Result of an ingest run, tracking what was processed."""
    run_id: str
    context: str
    started_at: datetime
    finished_at: datetime
    new_doc_ids: list[str]      # Docs ingested for first time
    updated_doc_ids: list[str]  # Docs re-ingested due to changes
    new_chunk_ids: list[str]    # All chunks created this run
    skipped_doc_ids: list[str]  # Docs skipped (unchanged)
    error_doc_ids: list[str]    # Docs that failed
    stats: dict                 # {files_scanned, total_chunks, etc.}


def ingest(config: AppConfig, *, ollama_host_override: str | None = None) -> dict:
    index_dir = config.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "hybrid.db"
    chroma_dir = index_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    lock_path = index_dir / "hybrid.db.lock"
    try:
        with Lock(lock_path, timeout=60):
            storage = Storage(db_path)
            storage.ensure_schema()

            ollama_host = ollama_host_override or config.ollama_host
            embedder = OllamaEmbedder(ollama_host, config.embedding_model)
            vectors = VectorStore(chroma_dir)

            stats = {"documents": 0, "chunks": 0, "skipped": 0}
            started_at = now_iso()
            run_id = sha256_text(started_at)

            for source in config.sources:
                if source.type == "repo":
                    _ingest_repo(source, storage, embedder, vectors, stats)
                elif source.type == "chat":
                    _ingest_chat(source, storage, embedder, vectors, stats)

            storage.record_run(run_id, started_at, dump_json(stats))
            storage.close()
            return stats
    except LockException as exc:
        raise RuntimeError(
            "Ingest lock is held by another process. Only one ingest can run at a time."
        ) from exc


def _conditional_embed(
    texts: list[str],
    storage: Storage,
    embedder: OllamaEmbedder,
    vectors: VectorStore,
    stats: dict,
    rechunk_only: bool,
) -> list[list[float]]:
    """
    Generate embeddings with optional reuse optimization.

    When rechunk_only=True:
    - Compute chunk_key for each text
    - Lookup existing chunks in SQLite
    - Reuse embeddings from Chroma if text unchanged
    - Only generate new embeddings for new/changed chunks
    - Update stats['embeddings_reused'] and stats['embeddings_new']

    When rechunk_only=False:
    - Generate all embeddings normally
    """
    if not rechunk_only:
        new_embeddings = embedder.embed(texts)
        stats["embeddings_new"] += len(new_embeddings)
        return new_embeddings

    # Rechunk optimization path
    embeddings: list[list[float]] = []
    texts_to_embed: list[str] = []
    text_to_embed_indices: list[int] = []

    for i, text in enumerate(texts):
        key = chunk_key(text)
        existing = storage.lookup_chunk_by_key(key)

        if existing and existing["text"] == text:
            # Reuse existing embedding
            chunk_id = existing["chunk_id"]
            result = vectors.get_embeddings([chunk_id])
            if result["ids"] and result["embeddings"]:
                embeddings.append(result["embeddings"][0])
                stats["embeddings_reused"] += 1
            else:
                # Fallback: chunk exists but no embedding found
                texts_to_embed.append(text)
                text_to_embed_indices.append(i)
        else:
            # Need new embedding
            texts_to_embed.append(text)
            text_to_embed_indices.append(i)

    # Generate new embeddings in batch
    if texts_to_embed:
        new_embeddings = embedder.embed(texts_to_embed)
        stats["embeddings_new"] += len(new_embeddings)

        # Merge reused and new embeddings in correct order
        final_embeddings: list[list[float] | None] = [None] * len(texts)

        # Place reused embeddings
        reused_idx = 0
        for i in range(len(texts)):
            if i not in text_to_embed_indices:
                final_embeddings[i] = embeddings[reused_idx]
                reused_idx += 1

        # Place new embeddings
        for idx, new_emb in zip(text_to_embed_indices, new_embeddings):
            final_embeddings[idx] = new_emb

        return final_embeddings  # type: ignore

    return embeddings


def _ingest_repo(source: SourceConfig, storage: Storage, embedder: OllamaEmbedder, vectors: VectorStore, stats: dict) -> None:
    for path in walk_files(source.path):
        text = read_text_utf8(path)
        if text is None:
            print(f"Warning: could not decode {path} as UTF-8, skipping.")
            continue
        doc_id = sha256_text(f"repo|{normalized_path(path)}")
        updated_at = iso_from_mtime(path)
        content_hash = sha256_text(text)
        existing = storage.get_document(doc_id)
        if existing and existing["content_hash"] == content_hash and existing["updated_at"] == updated_at:
            stats["skipped"] += 1
            continue

        chunk_ids = storage.delete_chunks_for_doc(doc_id) if existing else []
        if chunk_ids:
            vectors.delete(chunk_ids)

        meta = {
            "path": normalized_path(path),
            "ext": path.suffix.lower(),
            "mtime": updated_at,
            "repo": source.name,
        }
        storage.upsert_document(
            doc_id=doc_id,
            source_type="repo",
            source_uri=normalized_path(path),
            project=None,
            repo=source.name,
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
            ck = chunk_key(chunk.text)
            cmeta = {
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "path": normalized_path(path),
                "repo": source.name,
            }
            chunk_rows.append(
                (
                    chunk_id,
                    doc_id,
                    "repo",
                    None,
                    source.name,
                    chunk.ordinal,
                    chunk.text,
                    updated_at,
                    dump_json(cmeta),
                    ck,  # chunk_key
                )
            )
            fts_rows.append((chunk_id, chunk.text))
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append(
                {
                    "source_type": "repo",
                    "repo": source.name,
                    "doc_id": doc_id,
                    "ordinal": chunk.ordinal,
                    "path": normalized_path(path),
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                }
            )

        embeddings = embedder.embed(docs)
        storage.upsert_chunks(chunk_rows)
        storage.upsert_fts(fts_rows)
        vectors.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        stats["documents"] += 1
        stats["chunks"] += len(chunks)


def _ingest_chat(source: SourceConfig, storage: Storage, embedder: OllamaEmbedder, vectors: VectorStore, stats: dict) -> None:
    for path in Path(source.path).glob("*.json"):
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
        project = str(raw.get("project", source.project or ""))
        exported_at = str(raw.get("exported_at") or iso_from_mtime(path))
        doc_id = conversation_id

        lines = []
        for i, msg in enumerate(messages):
            role = str(msg.get("role", "unknown"))
            text = str(msg.get("text", "")).strip()
            lines.append(f"[{i:04d}] {role}: {text}")
        canonical_text = "\n".join(lines)
        content_hash = sha256_text(canonical_text)

        existing = storage.get_document(doc_id)
        if existing and existing["content_hash"] == content_hash and existing["updated_at"] == exported_at:
            stats["skipped"] += 1
            continue

        chunk_ids = storage.delete_chunks_for_doc(doc_id) if existing else []
        if chunk_ids:
            vectors.delete(chunk_ids)

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
            ck = chunk_key(chunk.text)
            cmeta = {
                "doc_id": doc_id,
                "ordinal": chunk.ordinal,
                "msg_start": chunk.msg_start,
                "msg_end": chunk.msg_end,
                "roles_present": chunk.roles_present or [],
                "project": project,
                "title": title,
            }
            chunk_rows.append(
                (
                    chunk_id,
                    doc_id,
                    "chat",
                    project,
                    None,
                    chunk.ordinal,
                    chunk.text,
                    exported_at,
                    dump_json(cmeta),
                    ck,  # chunk_key
                )
            )
            fts_rows.append((chunk_id, chunk.text))
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append(
                {
                    "source_type": "chat",
                    "project": project,
                    "doc_id": doc_id,
                    "ordinal": chunk.ordinal,
                    "msg_start": chunk.msg_start,
                    "msg_end": chunk.msg_end,
                    "title": title,
                }
            )

        embeddings = embedder.embed(docs)
        storage.upsert_chunks(chunk_rows)
        storage.upsert_fts(fts_rows)
        vectors.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        stats["documents"] += 1
        stats["chunks"] += len(chunks)


def ingest_context(
    ctx: ContextConfig,
    *,
    ollama_host_override: str | None = None,
    rechunk_only: bool = False,
    embed_provider: str | None = None,
    rebuild_index: bool = False,
) -> IngestRunResult:
    """
    Ingest all sources from a context.

    Uses context.index paths for storage.
    Applies context.weights for source-type prioritization.
    Uses fingerprinting for incremental ingest.

    When rechunk_only=True, rechunks files and reuses embeddings where possible
    (optimization for when only chunking strategy changed).

    Args:
        embed_provider: Override embedding provider (ollama|openai)
        rebuild_index: Force rebuild when switching providers
    """
    import uuid
    from datetime import timezone
    from .embedding_providers import get_provider
    from .index_meta import IndexMeta, read_index_meta, write_index_meta
    from .ingest_log import log_run_start, log_run_end
    from .util import now_iso

    db_path = ctx.index.sqlite_path
    chroma_dir = ctx.index.chroma_dir

    db_path.parent.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID and setup logging
    run_id = str(uuid.uuid4())
    log_path = db_path.parent.parent / ctx.name / "ingest_runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = db_path.parent / "hybrid.db.lock"
    try:
        with Lock(lock_path, timeout=60):
            storage = Storage(db_path)
            storage.ensure_schema()

            # Log sources being ingested
            sources = [str(p) for p in ctx.includes.repos]
            sources.extend([str(p) for p in ctx.includes.chat_roots])
            if ctx.includes.codex_session_roots:
                sources.extend([str(p) for p in ctx.includes.codex_session_roots])
            log_run_start(log_path, run_id, sources=sources)

            # Get provider with precedence: CLI > context.json > env > default
            env_provider = os.getenv("CHINVEX_EMBED_PROVIDER")
            ollama_host = ollama_host_override or ctx.ollama.base_url
            provider = get_provider(
                cli_provider=embed_provider,
                context_config=None,  # TODO: read from ctx config
                env_provider=env_provider,
                ollama_host=ollama_host
            )

            # Check index metadata for dimension compatibility
            index_dir = db_path.parent
            meta_path = index_dir / "meta.json"
            existing_meta = read_index_meta(meta_path)

            if existing_meta:
                # Validate dimensions match
                provider_name = provider.__class__.__name__.replace("Provider", "").lower()
                if not existing_meta.matches_provider(
                    provider_name,
                    provider.model_name,
                    provider.dimensions
                ):
                    if not rebuild_index:
                        raise RuntimeError(
                            f"Dimension mismatch: index uses {existing_meta.embedding_provider} "
                            f"({existing_meta.embedding_dimensions}D) but provider is "
                            f"{provider.model_name} ({provider.dimensions}D). "
                            f"Use --rebuild-index to switch providers."
                        )
                    # Rebuild: clear index
                    log.warning("Rebuilding index due to provider change")
                    # TODO: implement full index rebuild (clear SQLite + Chroma)
            else:
                # Create meta.json for new index
                meta = IndexMeta(
                    schema_version=2,
                    embedding_provider=provider.__class__.__name__.replace("Provider", "").lower(),
                    embedding_model=provider.model_name,
                    embedding_dimensions=provider.dimensions,
                    created_at=now_iso()
                )
                write_index_meta(meta_path, meta)

            # Use provider for embedding (wrap in OllamaEmbedder-compatible interface if needed)
            # For now, keep using OllamaEmbedder directly
            # TODO: refactor to use provider.embed() directly
            embedding_model = ctx.ollama.embed_model
            fallback_host = "http://127.0.0.1:11434" if ollama_host != "http://127.0.0.1:11434" else None
            embedder = OllamaEmbedder(ollama_host, embedding_model, fallback_host=fallback_host)
            vectors = VectorStore(chroma_dir)

            # Track results for IngestRunResult
            started_at = datetime.now(timezone.utc)
            new_doc_ids: list[str] = []
            updated_doc_ids: list[str] = []
            new_chunk_ids: list[str] = []
            skipped_doc_ids: list[str] = []
            error_doc_ids: list[str] = []
            stats = {
                "documents": 0,
                "chunks": 0,
                "skipped": 0,
                "embeddings_reused": 0,
                "embeddings_new": 0,
            }

            # Create tracking dict to pass to helper functions
            tracking = {
                "new_doc_ids": new_doc_ids,
                "updated_doc_ids": updated_doc_ids,
                "new_chunk_ids": new_chunk_ids,
                "skipped_doc_ids": skipped_doc_ids,
                "error_doc_ids": error_doc_ids,
            }

            try:
                # Ingest repos
                for repo_path in ctx.includes.repos:
                    if not repo_path.exists():
                        print(f"Warning: repo path {repo_path} does not exist, skipping.")
                        continue
                    _ingest_repo_from_context(
                        ctx, repo_path, storage, embedder, vectors, stats, tracking, rechunk_only
                    )

                # Ingest chat roots
                for chat_root in ctx.includes.chat_roots:
                    if not chat_root.exists():
                        print(f"Warning: chat_root {chat_root} does not exist, skipping.")
                        continue
                    _ingest_chat_from_context(
                        ctx, chat_root, storage, embedder, vectors, stats, tracking, rechunk_only
                    )

                # Ingest Codex sessions
                if ctx.includes.codex_session_roots and ctx.codex_appserver and ctx.codex_appserver.enabled:
                    appserver_url = ctx.codex_appserver.base_url
                    _ingest_codex_sessions_from_context(
                        ctx, appserver_url, storage, embedder, vectors, stats, tracking, rechunk_only
                    )

                # TODO: Ingest note_roots (post-P0)

                finished_at = datetime.now(timezone.utc)
                storage.record_run(run_id, now_iso(), dump_json(stats))

                # Auto-archive hook (before closing storage)
                if ctx.archive and ctx.archive.enabled and ctx.archive.auto_archive_on_ingest:
                    from .archive import archive_old_documents

                    archived_count = archive_old_documents(
                        storage,
                        age_threshold_days=ctx.archive.age_threshold_days,
                        dry_run=False
                    )

                    if archived_count > 0:
                        print(f"Archived {archived_count} docs (older than {ctx.archive.age_threshold_days}d)")

                storage.close()

                result = IngestRunResult(
                    run_id=run_id,
                    context=ctx.name,
                    started_at=started_at,
                    finished_at=finished_at,
                    new_doc_ids=new_doc_ids,
                    updated_doc_ids=updated_doc_ids,
                    new_chunk_ids=new_chunk_ids,
                    skipped_doc_ids=skipped_doc_ids,
                    error_doc_ids=error_doc_ids,
                    stats=stats
                )

                # Log successful run
                log_run_end(
                    log_path,
                    run_id,
                    status="succeeded",
                    docs_seen=stats["documents"] + stats["skipped"],
                    docs_changed=stats["documents"],
                    chunks_new=stats["chunks"],
                    chunks_updated=0  # TODO: track updates properly
                )

                # Call post-ingest hook
                try:
                    post_ingest_hook(ctx, result)
                except Exception as e:
                    log.error(f"Post-ingest hook failed: {e}")
                    # Don't fail ingest on hook failure

                return result
            except Exception as e:
                # Log failed run
                log_run_end(log_path, run_id, status="failed", error=str(e))
                raise
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
    tracking: dict,
    rechunk_only: bool = False,
) -> None:
    """Ingest a single repo with fingerprinting."""
    for path in walk_files(repo_path, excludes=ctx.includes.repo_excludes):
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
            tracking["skipped_doc_ids"].append(doc_id)
            continue

        # Track if this is new or updated
        is_new_doc = fp is None

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
            ck = chunk_key(chunk.text)
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
                ck,  # chunk_key
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

        embeddings = _conditional_embed(docs, storage, embedder, vectors, stats, rechunk_only)
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

        # Track document and chunk IDs
        if is_new_doc:
            tracking["new_doc_ids"].append(doc_id)
        else:
            tracking["updated_doc_ids"].append(doc_id)
        tracking["new_chunk_ids"].extend(ids)

        stats["documents"] += 1
        stats["chunks"] += len(chunks)


def _ingest_chat_from_context(
    ctx: ContextConfig,
    chat_root: Path,
    storage: Storage,
    embedder: OllamaEmbedder,
    vectors: VectorStore,
    stats: dict,
    tracking: dict,
    rechunk_only: bool = False,
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
            ck = chunk_key(chunk.text)
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
                ck,  # chunk_key
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

        embeddings = _conditional_embed(docs, storage, embedder, vectors, stats, rechunk_only)
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


def _ingest_codex_sessions_from_context(
    ctx: ContextConfig,
    appserver_url: str,
    storage: Storage,
    embedder: OllamaEmbedder,
    vectors: VectorStore,
    stats: dict,
    tracking: dict,
    rechunk_only: bool = False,
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
        chunks = chunk_conversation(conversation_doc)

        chunk_rows = []
        fts_rows = []
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for chunk in chunks:
            chunk_id = sha256_text(f"{doc_id}|{chunk.ordinal}|{sha256_text(chunk.text)}")
            ck = chunk_key(chunk.text)
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
                ck,  # chunk_key
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

        embeddings = _conditional_embed(docs, storage, embedder, vectors, stats, rechunk_only)
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
