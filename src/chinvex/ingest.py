from __future__ import annotations

import json
from pathlib import Path

from portalocker import Lock, LockException

from .chunking import chunk_chat, chunk_repo
from .config import AppConfig, SourceConfig
from .context import ContextConfig
from .embed import OllamaEmbedder
from .storage import Storage
from .util import dump_json, iso_from_mtime, normalized_path, now_iso, read_text_utf8, sha256_text, walk_files
from .vectors import VectorStore


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
