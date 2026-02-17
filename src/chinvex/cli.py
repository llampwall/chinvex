from __future__ import annotations

import os
from pathlib import Path

import typer

from .config import ConfigError, load_config
from .context_cli import create_context, get_contexts_root, get_indexes_root, list_contexts_cli
from .ingest import ingest
from .search import search
from .util import in_venv


# === Archive helpers for _archive context ===

def _extract_description_from_dir(dir_path: Path) -> str:
    """
    Extract description from a directory using fallback chain:
    1. docs/memory/STATE.md -> "Current Objective" line
    2. README.md -> first non-empty paragraph
    Returns empty string if nothing found.
    """
    # Try STATE.md first
    state_md = dir_path / "docs" / "memory" / "STATE.md"
    if state_md.exists():
        try:
            content = state_md.read_text(encoding="utf-8")
            for line in content.splitlines():
                # Look for "Current Objective:" or similar
                if "current objective" in line.lower():
                    # Extract the value after the colon
                    if ":" in line:
                        return line.split(":", 1)[1].strip()
                    # Or if it's a header, get the next non-empty line
        except Exception:
            pass

    # Try README.md
    readme_md = dir_path / "README.md"
    if readme_md.exists():
        try:
            content = readme_md.read_text(encoding="utf-8")
            lines = content.splitlines()
            # Skip title (usually first # line) and find first paragraph
            in_paragraph = False
            paragraph_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip empty lines and headers before finding content
                if not stripped:
                    if in_paragraph and paragraph_lines:
                        break  # End of first paragraph
                    continue
                if stripped.startswith("#"):
                    if in_paragraph and paragraph_lines:
                        break
                    continue
                # Found content
                in_paragraph = True
                paragraph_lines.append(stripped)

            if paragraph_lines:
                # Limit to reasonable length
                desc = " ".join(paragraph_lines)
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                return desc
        except Exception:
            pass

    return ""


def _add_to_archive_context(name: str, description: str) -> None:
    """
    Add an entry to the _archive context.
    Creates the _archive context if it doesn't exist.
    Entry format: "[name] description" as a single chunk.
    """
    import json
    from datetime import datetime, timezone
    from hashlib import sha256

    from .context_cli import create_context_if_missing

    contexts_root = get_contexts_root()

    # Ensure _archive context exists
    archive_ctx_dir = contexts_root / "_archive"
    if not archive_ctx_dir.exists():
        create_context_if_missing("_archive", contexts_root)

    # Load _archive context to get index paths
    archive_config_path = archive_ctx_dir / "context.json"
    archive_config = json.loads(archive_config_path.read_text(encoding="utf-8"))

    db_path = Path(archive_config["index"]["sqlite_path"])
    chroma_dir = Path(archive_config["index"]["chroma_dir"])

    # Create chunk content in parseable format
    chunk_content = f"[{name}] {description}" if description else f"[{name}] (no description)"

    # Generate IDs
    doc_id = f"archive:{name}"
    chunk_id = sha256(f"{doc_id}:0".encode()).hexdigest()[:16]

    # Insert into SQLite
    from .storage import Storage
    storage = Storage(db_path)
    storage.ensure_schema()

    now = datetime.now(timezone.utc).isoformat()

    # Check if entry already exists, update if so
    storage.conn.execute(
        "DELETE FROM chunks WHERE doc_id = ?",
        (doc_id,)
    )
    storage.conn.execute(
        "DELETE FROM documents WHERE doc_id = ?",
        (doc_id,)
    )

    # Insert document (schema: doc_id, source_type, source_uri, project, repo, title, updated_at, ...)
    storage.conn.execute(
        """INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (doc_id, "archive_entry", f"archive://{name}", name, now)
    )

    # Insert chunk (schema: chunk_id, doc_id, source_type, project, repo, ordinal, text, updated_at, ...)
    storage.conn.execute(
        """INSERT INTO chunks (chunk_id, doc_id, source_type, ordinal, text, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chunk_id, doc_id, "archive_entry", 0, chunk_content, now)
    )

    # Insert into FTS
    storage.conn.execute(
        "INSERT INTO chunks_fts(rowid, text) VALUES ((SELECT rowid FROM chunks WHERE chunk_id = ?), ?)",
        (chunk_id, chunk_content)
    )

    storage.conn.commit()
    storage.close()

    # Insert into Chroma for vector search
    from .vectors import VectorStore
    vectors = VectorStore(chroma_dir)

    # Get embedding - skip if it fails (FTS still works)
    try:
        from .embeddings import get_embedding
        embedding = get_embedding(chunk_content)
        vectors.add(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[{"doc_id": doc_id, "source_type": "archive_entry"}]
        )
    except Exception:
        # If embedding fails, skip vector indexing (search still works via FTS)
        pass


def _delete_context(name: str) -> bool:
    """
    Delete a context and its index directory.
    Returns True if deleted, False if not found.
    Raises PermissionError if files are locked.
    """
    import shutil
    from .storage import Storage

    contexts_root = get_contexts_root()
    indexes_root = get_indexes_root()

    ctx_dir = contexts_root / name
    idx_dir = indexes_root / name

    if not ctx_dir.exists():
        return False

    # Force close any open database connections
    Storage.force_close_global_connection()

    # Delete context directory
    shutil.rmtree(ctx_dir)

    # Delete index directory
    if idx_dir.exists():
        shutil.rmtree(idx_dir)

    return True

app = typer.Typer(add_completion=False, help="chinvex: hybrid retrieval index CLI")

# Add context subcommand group
context_app = typer.Typer(help="Manage contexts")
app.add_typer(context_app, name="context")

# Add state subcommand group
state_app = typer.Typer(help="Manage context state and STATE.md")
app.add_typer(state_app, name="state")

# Add state note subcommand group
state_note_app = typer.Typer(help="Manage state annotations")
state_app.add_typer(state_note_app, name="note")

# Add watch subcommand group
watch_app = typer.Typer(help="Manage watch queries")
app.add_typer(watch_app, name="watch")

# Add digest subcommand group
digest_app = typer.Typer(help="Generate digest reports")
app.add_typer(digest_app, name="digest")

# Add sync subcommand group
sync_app = typer.Typer(help="File watcher sync daemon")
app.add_typer(sync_app, name="sync")

# Add hook subcommand group
hook_app = typer.Typer(help="Git hook management")
app.add_typer(hook_app, name="hook")

# Add bootstrap subcommand group
bootstrap_app = typer.Typer(help="Bootstrap installation commands")
app.add_typer(bootstrap_app, name="bootstrap")


def _load_config(config_path: Path):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("ingest")
def ingest_cmd(
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to ingest"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    paths: str | None = typer.Option(None, "--paths", help="Comma-separated paths for delta ingest (optional)"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
    rechunk_only: bool = typer.Option(False, "--rechunk-only", help="Rechunk only, reuse embeddings when possible"),
    embed_provider: str | None = typer.Option(None, "--embed-provider", help="Embedding provider: ollama|openai"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Rebuild index (wipe and re-embed)"),
    repo: list[str] = typer.Option([], "--repo", help="Add repo path to context (can be repeated)"),
    chat_root: list[str] = typer.Option([], "--chat-root", help="Add chat root to context (can be repeated)"),
    no_write_context: bool = typer.Option(False, "--no-write-context", help="Ingest ad-hoc without mutating context.json"),
    no_claude_hook: bool = typer.Option(False, "--no-claude-hook", help="Skip Claude Code startup hook installation"),
    register_only: bool = typer.Option(False, "--register-only", help="Register paths in context.json without ingesting"),
    chinvex_depth: str = typer.Option("full", "--chinvex-depth", help="Ingestion depth: full, light, or index"),
    status: str = typer.Option("active", "--status", help="Lifecycle status: active, stable, or dormant"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags (e.g., 'python,ml,web')"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Validate new metadata fields
    if chinvex_depth not in ["full", "light", "index"]:
        typer.secho(f"Error: Invalid --chinvex-depth: {chinvex_depth}. Must be: full, light, or index", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if status not in ["active", "stable", "dormant"]:
        typer.secho(f"Error: Invalid --status: {status}. Must be: active, stable, or dormant", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Parse tags
    tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Handle --register-only: add paths to context.json without ingesting
    if register_only:
        if not context:
            typer.secho("Error: --register-only requires --context", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        if not repo and not chat_root:
            typer.secho("Error: --register-only requires --repo or --chat-root", fg=typer.colors.RED)
            raise typer.Exit(code=2)

        from .context_cli import create_context_if_missing

        contexts_root = get_contexts_root()

        # Validate paths exist
        for repo_path in repo:
            if not Path(repo_path).exists():
                typer.secho(f"Error: Repo path does not exist: {repo_path}", fg=typer.colors.RED)
                raise typer.Exit(code=2)
        for chat_path in chat_root:
            if not Path(chat_path).exists():
                typer.secho(f"Error: Chat root does not exist: {chat_path}", fg=typer.colors.RED)
                raise typer.Exit(code=2)

        # Create repo metadata objects
        repo_metadata = []
        if repo:
            for repo_path in repo:
                repo_metadata.append({
                    "path": repo_path,
                    "chinvex_depth": chinvex_depth,
                    "status": status,
                    "tags": tags_list
                })

        # Create context if missing, or update with new paths
        create_context_if_missing(
            context,
            contexts_root,
            repos=repo_metadata if repo_metadata else None,
            chat_roots=chat_root if chat_root else None
        )

        typer.secho(f"Registered paths in context '{context}' (no ingestion)", fg=typer.colors.GREEN)
        typer.echo(f"  depth={chinvex_depth}, status={status}, tags={tags_list}")
        if repo:
            for r in repo:
                typer.echo(f"  repo: {r}")
        if chat_root:
            for c in chat_root:
                typer.echo(f"  chat_root: {c}")
        return

    if context:
        # Validate repo paths exist
        for repo_path in repo:
            if not Path(repo_path).exists():
                typer.secho(f"Error: Repo path does not exist: {repo_path}", fg=typer.colors.RED)
                raise typer.Exit(code=2)

        # Validate chat roots exist
        for chat_path in chat_root:
            if not Path(chat_path).exists():
                typer.secho(f"Error: Chat root does not exist: {chat_path}", fg=typer.colors.RED)
                raise typer.Exit(code=2)

        # New context-based ingestion
        from .context import load_context
        from .context_cli import create_context_if_missing
        from .ingest import ingest_context

        contexts_root = get_contexts_root()

        # Auto-create context if needed (unless --no-write-context)
        if no_write_context:
            # For --no-write-context, create context in memory only
            from .context import ContextConfig, ContextIncludes, ContextIndex, EmbeddingConfig, OllamaConfig, RepoMetadata
            from datetime import datetime, timezone

            indexes_root = get_contexts_root().parent / "indexes"
            index_dir = indexes_root / context
            index_dir.mkdir(parents=True, exist_ok=True)

            # Initialize database if needed
            from .storage import Storage
            db_path = index_dir / "hybrid.db"
            if not db_path.exists():
                storage = Storage(db_path)
                storage.ensure_schema()
                storage.close()

            # Initialize Chroma if needed
            from .vectors import VectorStore
            chroma_dir = index_dir / "chroma"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            if not (chroma_dir / "chroma.sqlite3").exists():
                vectors = VectorStore(chroma_dir)

            # Parse tags
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            now = datetime.now(timezone.utc).isoformat()
            ctx = ContextConfig(
                schema_version=1,
                name=context,
                aliases=[],
                includes=ContextIncludes(
                    repos=[RepoMetadata(
                        path=Path(r).resolve(),
                        chinvex_depth=chinvex_depth,
                        status=status,
                        tags=tag_list
                    ) for r in repo],
                    chat_roots=[Path(c).resolve() for c in chat_root],
                    codex_session_roots=[],
                    note_roots=[]
                ),
                index=ContextIndex(
                    sqlite_path=db_path,
                    chroma_dir=chroma_dir
                ),
                weights={"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
                ollama=OllamaConfig(
                    base_url="http://skynet:11434",
                    embed_model="mxbai-embed-large"
                ),
                embedding=EmbeddingConfig(
                    provider="openai",
                    model="text-embedding-3-small"
                ),
                created_at=now,
                updated_at=now
            )
        else:
            # Normal path: create/update context.json and load it
            # Transform repo paths to proper dict format with metadata
            repos_with_metadata = None
            if repo:
                tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
                repos_with_metadata = [{
                    "path": r,
                    "chinvex_depth": chinvex_depth,
                    "status": status,
                    "tags": tag_list
                } for r in repo]

            create_context_if_missing(
                context,
                contexts_root,
                repos=repos_with_metadata,
                chat_roots=chat_root if chat_root else None
            )
            ctx = load_context(context, contexts_root)

        # Check if delta ingest (--paths) or full ingest
        if paths:
            from .ingest import ingest_delta

            # Parse paths
            path_list = [Path(p.strip()) for p in paths.split(",") if p.strip()]
            typer.echo(f"Delta ingest: {len(path_list)} files for context {context}")

            result = ingest_delta(
                ctx,
                path_list,
                contexts_root=contexts_root,
                ollama_host_override=ollama_host,
                embed_provider=embed_provider
            )

            typer.secho(f"Delta ingestion complete for context '{context}':", fg=typer.colors.GREEN)
            typer.echo(f"  Files processed: {result.stats.get('files_processed', 0)}")
            typer.echo(f"  Documents: {result.stats['documents']}")
            typer.echo(f"  Chunks: {result.stats['chunks']}")
            typer.echo(f"  Skipped: {result.stats['skipped']}")
        else:
            result = ingest_context(
                ctx,
                contexts_root=contexts_root,
                ollama_host_override=ollama_host,
                rechunk_only=rechunk_only,
                embed_provider=embed_provider,
                rebuild_index=rebuild_index
            )

            typer.secho(f"Ingestion complete for context '{context}':", fg=typer.colors.GREEN)
            typer.echo(f"  Documents: {result.stats['documents']}")
            typer.echo(f"  Chunks: {result.stats['chunks']}")
            typer.echo(f"  Skipped: {result.stats['skipped']}")
            if 'embeddings_reused' in result.stats:
                typer.echo(f"  Embeddings: {result.stats['embeddings_reused']} reused, {result.stats['embeddings_new']} new")

        # Install startup hooks in repos after successful ingestion
        if not no_claude_hook:
            from .hook_installer import install_startup_hook

            repos = [r.path for r in ctx.includes.repos]
            for repo_path in repos:
                if repo_path.exists():
                    success = install_startup_hook(repo_path, context)
                    if success:
                        typer.echo(f"Installed startup hook in {repo_path}")
                    else:
                        typer.secho(f"Warning: Could not install hook in {repo_path}", fg=typer.colors.YELLOW, err=True)
    else:
        # Old config-based ingestion (deprecated)
        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)

        cfg = _load_config(config)
        stats = ingest(cfg, ollama_host_override=ollama_host)
        typer.secho("Ingestion complete:", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {stats['documents']}")
        typer.echo(f"  Chunks: {stats['chunks']}")
        typer.echo(f"  Skipped: {stats['skipped']}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to search (deprecated for multi-context)"),
    contexts: str | None = typer.Option(None, "--contexts", help="Comma-separated context names"),
    all_contexts: bool = typer.Option(False, "--all", help="Search all contexts"),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated contexts to exclude (with --all)"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    k: int = typer.Option(8, "--k", help="Top K results"),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum score threshold"),
    source: str = typer.Option("all", "--source", help="all|repo|chat|codex_session"),
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repo"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
    no_recency: bool = typer.Option(False, "--no-recency", help="Disable recency decay"),
    allow_mixed_embeddings: bool = typer.Option(False, "--allow-mixed-embeddings", help="Allow mixed embedding providers (P6+)"),
    rerank: bool = typer.Option(False, "--rerank", help="Enable reranking for this query"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if source not in {"all", "repo", "chat", "codex_session"}:
        raise typer.BadParameter("source must be one of: all, repo, chat, codex_session")

    # Check for --allow-mixed-embeddings flag
    if allow_mixed_embeddings:
        typer.secho(
            "Error: Mixed-space embedding merge is not yet supported. "
            "This feature is planned for P6+. "
            "For now, ensure all contexts use the same embedding provider.",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Determine search mode: multi-context, single-context, or legacy config
    if all_contexts or contexts:
        # Multi-context search
        from .context import list_contexts as list_contexts_func
        from .search import search_multi_context

        contexts_root = get_contexts_root()

        if all_contexts:
            ctx_list = "all"
            if exclude:
                # Filter out excluded contexts
                all_ctx = [c.name for c in list_contexts_func(contexts_root)]
                excluded = [x.strip() for x in exclude.split(",")]
                ctx_list = [c for c in all_ctx if c not in excluded]
        elif contexts:
            ctx_list = [x.strip() for x in contexts.split(",")]
        else:
            ctx_list = "all"

        # Note: project/repo filters not supported in multi-context mode
        if project or repo:
            typer.secho("Warning: --project and --repo filters not supported in multi-context search", fg=typer.colors.YELLOW)

        results = search_multi_context(
            contexts=ctx_list,
            query=query,
            k=k,
            min_score=min_score,
            source=source,
            ollama_host=ollama_host,
            recency_enabled=not no_recency,
            allow_mixed_embeddings=allow_mixed_embeddings,
            rerank=rerank,
        )

        if not results:
            typer.echo("No results found.")
            return

        # Print results with context tags
        typer.secho(f"\nSearched contexts: {ctx_list if isinstance(ctx_list, list) else 'all'}", fg=typer.colors.BLUE)
        typer.secho(f"Found {len(results)} results\n", fg=typer.colors.GREEN)

        # Score distribution stats
        if results:
            score_min = min(r.score for r in results)
            score_max = max(r.score for r in results)
            context_counts = {}
            for r in results:
                context_counts[r.context] = context_counts.get(r.context, 0) + 1
            typer.echo(f"Score range: {score_min:.3f} - {score_max:.3f}")
            typer.echo(f"Results by context: {context_counts}\n")

        for i, result in enumerate(results, 1):
            typer.secho(f"[{i}] [{result.context}] {result.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"Score: {result.score:.3f} | Type: {result.source_type}")
            typer.echo(f"Citation: {result.citation}")
            typer.echo(f"Snippet: {result.snippet}\n")

    elif context:
        # New context-based search
        from .context import load_context
        from .search import search_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)

        results = search_context(
            ctx,
            query,
            k=k,
            min_score=min_score,
            source=source,
            project=project,
            repo=repo,
            ollama_host_override=ollama_host,
            recency_enabled=not no_recency,
            rerank=rerank,
        )

        if not results:
            typer.echo("No results found.")
            return

        for i, result in enumerate(results, 1):
            typer.secho(f"\n[{i}] {result.title}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"Score: {result.score:.3f} | Type: {result.source_type}")
            typer.echo(f"Citation: {result.citation}")
            typer.echo(f"Snippet: {result.snippet}")
    elif config:
        # Old config-based search (deprecated)
        typer.secho("Warning: --config is deprecated. Use --context instead.", fg=typer.colors.YELLOW)

        cfg = _load_config(config)
        results = search(
            cfg,
            query,
            k=k,
            min_score=min_score,
            source=source,
            project=project,
            repo=repo,
            ollama_host_override=ollama_host,
        )
        if not results:
            typer.echo("No results.")
            raise typer.Exit(code=0)

        for idx, result in enumerate(results, start=1):
            typer.echo(f"{idx}. score={result.score:.3f} source={result.source_type}")
            typer.echo(f"   {result.title}")
            typer.echo(f"   {result.citation}")
            typer.echo(f"   {result.snippet}")
    else:
        # No context flags provided
        typer.secho("Error: Must specify --context, --contexts, --all, or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)


@context_app.command("create")
def context_create_cmd(
    name: str = typer.Argument(..., help="Context name"),
    idempotent: bool = typer.Option(False, "--idempotent", help="No-op if context already exists"),
) -> None:
    """Create a new context."""
    if idempotent:
        from .context_cli import create_context_if_missing, get_contexts_root
        contexts_root = get_contexts_root()
        ctx_dir = contexts_root / name
        if ctx_dir.exists():
            typer.echo(f"Context '{name}' already exists (idempotent)")
            return
        create_context_if_missing(name, contexts_root)
        typer.secho(f"Created context: {name}", fg=typer.colors.GREEN)
    else:
        create_context(name)


@context_app.command("list")
def context_list_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all contexts."""
    if json_output:
        import json
        from .context import list_contexts
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        output = [
            {
                "name": ctx.name,
                "aliases": ctx.aliases,
                "updated_at": ctx.updated_at,
            }
            for ctx in contexts
        ]
        typer.echo(json.dumps(output, indent=2))
    else:
        list_contexts_cli()


@context_app.command("sync-metadata-from-strap")
def context_sync_metadata_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    registry: Path | None = typer.Option(None, "--registry", help="Path to registry.json (default: P:/software/_strap/registry.json)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """
    Sync repo metadata from strap registry.json to context.json.

    Updates status, tags, and chinvex_depth for repos in the context
    to match values in strap's registry.
    """
    from .context_cli import sync_metadata_from_strap

    try:
        result = sync_metadata_from_strap(context, registry)

        if json_output:
            import json
            typer.echo(json.dumps(result, indent=2))
        else:
            if result["updated"]:
                typer.secho(f"Updated {len(result['updated'])} repo(s):", fg=typer.colors.GREEN)
                for item in result["updated"]:
                    typer.echo(f"  {item['path']}")
                    if item['old']['depth'] != item['new']['depth']:
                        typer.echo(f"    depth: {item['old']['depth']} -> {item['new']['depth']}")
                    if item['old']['status'] != item['new']['status']:
                        typer.echo(f"    status: {item['old']['status']} -> {item['new']['status']}")
                    if item['old']['tags'] != item['new']['tags']:
                        typer.echo(f"    tags: {item['old']['tags']} -> {item['new']['tags']}")
            else:
                typer.echo("No changes needed (metadata already in sync)")

            if result["not_found"]:
                typer.secho(f"\nWarning: {len(result['not_found'])} repo(s) not found in registry:", fg=typer.colors.YELLOW)
                for path in result["not_found"]:
                    typer.echo(f"  {path}")

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@context_app.command("exists")
def context_exists_cmd(
    name: str = typer.Argument(..., help="Context name to check"),
) -> None:
    """Check if a context exists. Exits 0 if exists, 1 if not."""
    contexts_root = get_contexts_root()
    ctx_dir = contexts_root / name
    if ctx_dir.exists() and (ctx_dir / "context.json").exists():
        typer.echo(f"Context '{name}' exists")
        raise typer.Exit(code=0)
    else:
        typer.echo(f"Context '{name}' does not exist")
        raise typer.Exit(code=1)


@context_app.command("rename")
def context_rename_cmd(
    old_name: str = typer.Argument(..., help="Current context name"),
    new_name: str = typer.Option(..., "--to", help="New context name"),
) -> None:
    """Rename a context."""
    import json
    import shutil
    from .context_cli import get_indexes_root

    contexts_root = get_contexts_root()
    indexes_root = get_indexes_root()

    old_ctx_dir = contexts_root / old_name
    new_ctx_dir = contexts_root / new_name
    old_idx_dir = indexes_root / old_name
    new_idx_dir = indexes_root / new_name

    # Validate source exists
    if not old_ctx_dir.exists():
        typer.secho(f"Context '{old_name}' does not exist", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Validate target doesn't exist
    if new_ctx_dir.exists():
        typer.secho(f"Context '{new_name}' already exists", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Rename context directory
        shutil.move(str(old_ctx_dir), str(new_ctx_dir))

        # Rename index directory if it exists
        if old_idx_dir.exists():
            shutil.move(str(old_idx_dir), str(new_idx_dir))
    except PermissionError as e:
        typer.secho(
            f"Cannot rename: files are locked. Stop any running processes using this context first.",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Update context.json
    from .util import backup_context_json
    context_file = new_ctx_dir / "context.json"
    if context_file.exists():
        ctx_data = json.loads(context_file.read_text(encoding="utf-8"))
        ctx_data["name"] = new_name
        # Update index paths
        if "index" in ctx_data:
            ctx_data["index"]["sqlite_path"] = str(new_idx_dir / "hybrid.db")
            ctx_data["index"]["chroma_dir"] = str(new_idx_dir / "chroma")
        backup_context_json(context_file)
        context_file.write_text(json.dumps(ctx_data, indent=2), encoding="utf-8")

    typer.secho(f"Renamed context '{old_name}' to '{new_name}'", fg=typer.colors.GREEN)


@context_app.command("remove-repo")
def context_remove_repo_cmd(
    context: str = typer.Argument(..., help="Context name"),
    repo: str = typer.Option(..., "--repo", help="Repo path to remove"),
    prune: bool = typer.Option(False, "--prune", help="Also delete indexed chunks from this repo"),
) -> None:
    """Remove a repo path from a context's configuration."""
    import json
    from .util import normalize_path_for_dedup

    contexts_root = get_contexts_root()
    ctx_dir = contexts_root / context
    context_file = ctx_dir / "context.json"

    if not context_file.exists():
        typer.secho(f"Context '{context}' does not exist", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Load context config
    ctx_data = json.loads(context_file.read_text(encoding="utf-8"))
    repos = ctx_data.get("includes", {}).get("repos", [])

    # Normalize the target path for matching
    target_normalized = normalize_path_for_dedup(repo)

    # Find and remove matching repo
    original_count = len(repos)
    filtered_repos = []
    for r in repos:
        # Handle both string and dict formats
        repo_path = r if isinstance(r, str) else r.get("path", "")
        if normalize_path_for_dedup(repo_path) != target_normalized:
            filtered_repos.append(r)
    repos = filtered_repos

    if len(repos) == original_count:
        typer.secho(f"Repo path not found in context '{context}': {repo}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Update and save
    from .util import backup_context_json
    ctx_data["includes"]["repos"] = repos
    backup_context_json(context_file)
    context_file.write_text(json.dumps(ctx_data, indent=2), encoding="utf-8")

    typer.secho(f"Removed repo from context '{context}': {repo}", fg=typer.colors.GREEN)

    # Handle --prune
    if prune:
        typer.secho(
            "Warning: --prune is not yet implemented. "
            "Indexed chunks from this repo remain in the database. "
            "Run 'chinvex ingest --context {context} --rebuild-index' to clean up.",
            fg=typer.colors.YELLOW
        )


@context_app.command("archive")
def context_archive_cmd(
    name: str = typer.Argument(..., help="Context name to archive"),
) -> None:
    """
    Archive an existing context to the _archive table of contents.

    Extracts name + description from the context's repos, adds an entry
    to the _archive context, then deletes the full context and index.
    """
    import json

    contexts_root = get_contexts_root()
    ctx_dir = contexts_root / name
    context_file = ctx_dir / "context.json"

    # Validate context exists
    if not context_file.exists():
        typer.secho(f"Context '{name}' does not exist", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Prevent archiving _archive itself
    if name == "_archive":
        typer.secho("Cannot archive the _archive context itself", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Load context config
    ctx_data = json.loads(context_file.read_text(encoding="utf-8"))
    repos = ctx_data.get("includes", {}).get("repos", [])

    # Extract description from first repo using fallback chain
    description = ""
    for repo_entry in repos:
        # Handle both string and dict formats
        repo_path = repo_entry if isinstance(repo_entry, str) else repo_entry.get("path", "")
        repo_dir = Path(repo_path)
        if repo_dir.exists():
            description = _extract_description_from_dir(repo_dir)
            if description:
                break

    # Add to _archive context
    try:
        _add_to_archive_context(name, description)
    except Exception as e:
        typer.secho(f"Failed to add to _archive: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Delete the full context
    try:
        _delete_context(name)
    except PermissionError:
        typer.secho(
            f"Cannot delete context: files are locked. "
            "Stop any running processes using this context first.",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    typer.secho(f"Archived context '{name}' to _archive", fg=typer.colors.GREEN)
    if description:
        typer.echo(f"  Description: {description[:60]}{'...' if len(description) > 60 else ''}")
    else:
        typer.echo("  Description: (none found)")


def _purge_context_data(
    ctx_name: str,
    contexts_root: Path,
) -> tuple[bool, str | None]:
    """
    Completely purge a context - deletes the context directory and its index directory.

    Returns:
        (success, error_message) tuple
    """
    context_dir = contexts_root / ctx_name

    if not context_dir.exists():
        return (False, f"Context '{ctx_name}' does not exist")

    try:
        _delete_context(ctx_name)
        return (True, None)
    except PermissionError as e:
        return (False, f"Permission denied for '{ctx_name}': {e}")
    except Exception as e:
        return (False, f"Error purging '{ctx_name}': {e}")


@context_app.command("purge")
def context_purge_cmd(
    name: str | None = typer.Argument(None, help="Context name to purge"),
    all_contexts: bool = typer.Option(False, "--all", help="Purge all contexts"),
) -> None:
    """
    Completely purge one or all contexts - deletes everything including registration.

    This deletes the entire context directory including:
    - Context configuration (context.json)
    - SQLite FTS5 index (hybrid.db)
    - ChromaDB vector embeddings (chroma/ directory)
    - Index metadata (meta.json)
    - Watch history log (watch_history.jsonl)
    - Digest cache (.digests/ directory)

    Examples:
        chinvex context purge allmind              # Purge one context completely
        chinvex context purge --all                # Purge all contexts completely
    """
    from .context import list_contexts
    from .storage import Storage

    contexts_root = get_contexts_root()

    # Validate arguments
    if not name and not all_contexts:
        typer.secho("Error: Must specify context name or --all", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if name and all_contexts:
        typer.secho("Error: Cannot specify both context name and --all", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Get list of contexts to purge
    if all_contexts:
        all_ctx = list_contexts(contexts_root)
        context_names = [ctx.name for ctx in all_ctx]

        if not context_names:
            typer.echo("No contexts found to purge.")
            raise typer.Exit(code=0)

        # Show all contexts that will be purged
        typer.echo(f"This will COMPLETELY DELETE {len(context_names)} context(s):")
        for ctx_name in context_names:
            typer.echo(f"  - {ctx_name}")
        typer.echo()
        typer.echo("WARNING: This deletes the entire context directory including configuration.")
        typer.echo()

        # Single confirmation prompt
        confirm = typer.confirm(
            f"Are you sure you want to completely purge ALL {len(context_names)} contexts?",
            default=False
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)
    else:
        context_names = [name]
        context_dir = contexts_root / name

        if not context_dir.exists():
            typer.secho(f"Context '{name}' does not exist", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        # Show what will be deleted
        typer.echo(f"This will COMPLETELY DELETE context '{name}':")
        typer.echo(f"  - Entire context directory: {context_dir}")
        typer.echo()
        typer.echo("WARNING: This deletes everything including configuration.")
        typer.echo()

        # Confirmation prompt
        confirm = typer.confirm("Are you sure you want to completely purge this context?", default=False)
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    # Force close any open database connections
    Storage.force_close_global_connection()

    # Purge each context
    purged_count = 0
    errors = []

    for ctx_name in context_names:
        success, error_msg = _purge_context_data(ctx_name, contexts_root)

        if success:
            purged_count += 1
            if all_contexts:
                typer.secho(f"[OK] Purged '{ctx_name}'", fg=typer.colors.GREEN)
            else:
                typer.secho(f"[OK] Purged context '{ctx_name}' completely", fg=typer.colors.GREEN)
        else:
            errors.append(error_msg)

    # Show summary for --all
    if all_contexts:
        typer.echo()
        typer.secho(f"Summary: Purged {purged_count}/{len(context_names)} context(s)", fg=typer.colors.GREEN)

    # Show any errors
    if errors:
        typer.echo()
        typer.secho("Errors encountered:", fg=typer.colors.RED)
        for error in errors:
            typer.echo(f"  [ERROR] {error}")
        raise typer.Exit(code=1)


@app.command("archive-unmanaged")
def archive_unmanaged_cmd(
    name: str = typer.Option(..., "--name", help="Name for the archive entry"),
    dir: Path = typer.Option(..., "--dir", help="Directory path to archive"),
    desc: str | None = typer.Option(None, "--desc", help="Description (auto-detected if not provided)"),
) -> None:
    """
    Add an unmanaged directory to the _archive table of contents.

    Use this for repos that were never managed as full contexts.
    If --desc is not provided, scans the directory for description
    using fallback chain: docs/memory/STATE.md -> README.md.
    """
    # Validate directory exists
    if not dir.exists():
        typer.secho(f"Directory does not exist: {dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Get description
    if desc:
        description = desc
    else:
        description = _extract_description_from_dir(dir)

    # Add to _archive context
    try:
        _add_to_archive_context(name, description)
    except Exception as e:
        typer.secho(f"Failed to add to _archive: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"Added '{name}' to _archive", fg=typer.colors.GREEN)
    if description:
        typer.echo(f"  Description: {description[:60]}{'...' if len(description) > 60 else ''}")
    else:
        typer.echo("  Description: (none found)")


@state_app.command("generate")
def state_generate_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    llm: bool = typer.Option(False, "--llm", help="Enable LLM consolidation (P1.5)"),
    since: str = typer.Option("24h", "--since", help="Time window (e.g., 24h, 7d)"),
) -> None:
    """Generate state.json and STATE.md."""
    from datetime import datetime, timedelta, timezone
    from .context import load_context
    from .hooks import post_ingest_hook
    from .ingest import IngestRunResult

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Parse since duration (simple implementation)
    # TODO: implement full duration parsing
    if since.endswith("h"):
        hours = int(since[:-1])
        since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    elif since.endswith("d"):
        days = int(since[:-1])
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        typer.secho(f"Invalid --since format: {since}. Use format like '24h' or '7d'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Create fake result for manual generation
    result = IngestRunResult(
        run_id=f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        context=context,
        started_at=since_dt,
        finished_at=datetime.now(timezone.utc),
        new_doc_ids=[],
        updated_doc_ids=[],
        new_chunk_ids=[],
        skipped_doc_ids=[],
        error_doc_ids=[],
        stats={}
    )

    try:
        post_ingest_hook(ctx, result)
        typer.secho(f"Generated STATE.md for context '{context}'", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error generating state: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@state_app.command("show")
def state_show_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
) -> None:
    """Print STATE.md to stdout."""
    contexts_root = get_contexts_root()
    md_path = contexts_root / context / "STATE.md"

    if not md_path.exists():
        typer.secho(f"No STATE.md found for context '{context}'", fg=typer.colors.RED)
        typer.echo(f"Run 'chinvex state generate --context {context}' to create it.")
        raise typer.Exit(code=1)

    typer.echo(md_path.read_text(encoding='utf-8'))


@state_note_app.command("add")
def state_note_add_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    note: str = typer.Argument(..., help="Annotation text"),
) -> None:
    """Add an annotation to state.json."""
    import json
    from datetime import datetime, timezone

    contexts_root = get_contexts_root()
    state_path = contexts_root / context / "state.json"

    if not state_path.exists():
        typer.secho(f"No state.json found for context '{context}'", fg=typer.colors.RED)
        typer.echo(f"Run 'chinvex state generate --context {context}' first.")
        raise typer.Exit(code=1)

    # Load existing state
    state_data = json.loads(state_path.read_text(encoding='utf-8'))

    # Add annotation
    if "annotations" not in state_data:
        state_data["annotations"] = []

    state_data["annotations"].append({
        "text": note,
        "added_at": datetime.now(timezone.utc).isoformat(),
    })

    # Save updated state
    state_path.write_text(json.dumps(state_data, indent=2), encoding='utf-8')
    typer.secho(f"Added annotation to {context}", fg=typer.colors.GREEN)


@watch_app.command("add")
def watch_add_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    id: str = typer.Option(..., "--id", help="Watch ID"),
    query: str = typer.Option(..., "--query", help="Search query to watch"),
    min_score: float = typer.Option(0.5, "--min-score", help="Minimum score threshold"),
) -> None:
    """Add a new watch query."""
    import json
    from datetime import datetime, timezone

    contexts_root = get_contexts_root()
    watch_path = contexts_root / context / "watch.json"

    # Load or create watch config
    if watch_path.exists():
        watch_data = json.loads(watch_path.read_text(encoding='utf-8'))
    else:
        watch_data = {"schema_version": 1, "watches": []}

    # Check for duplicate ID
    if any(w["id"] == id for w in watch_data["watches"]):
        typer.secho(f"Watch ID '{id}' already exists", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Add watch
    watch_data["watches"].append({
        "id": id,
        "query": query,
        "min_score": min_score,
        "enabled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    # Save
    watch_path.write_text(json.dumps(watch_data, indent=2), encoding='utf-8')
    typer.secho(f"Added watch '{id}'", fg=typer.colors.GREEN)


@watch_app.command("list")
def watch_list_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
) -> None:
    """List all watches for a context."""
    import json

    contexts_root = get_contexts_root()
    watch_path = contexts_root / context / "watch.json"

    if not watch_path.exists():
        typer.echo(f"No watches configured for {context}")
        return

    watch_data = json.loads(watch_path.read_text(encoding='utf-8'))

    if not watch_data.get("watches"):
        typer.echo(f"No watches configured for {context}")
        return

    for watch in watch_data["watches"]:
        status = "enabled" if watch.get("enabled", True) else "disabled"
        typer.secho(f"[{watch['id']}]", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"  Query: {watch['query']}")
        typer.echo(f"  Min score: {watch['min_score']}")
        typer.echo(f"  Status: {status}")
        typer.echo(f"  Created: {watch.get('created_at', 'unknown')}")


@watch_app.command("remove")
def watch_remove_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    id: str = typer.Option(..., "--id", help="Watch ID to remove"),
) -> None:
    """Remove a watch query."""
    import json

    contexts_root = get_contexts_root()
    watch_path = contexts_root / context / "watch.json"

    if not watch_path.exists():
        typer.secho(f"No watches found for {context}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    watch_data = json.loads(watch_path.read_text(encoding='utf-8'))

    # Find and remove watch
    original_count = len(watch_data["watches"])
    watch_data["watches"] = [w for w in watch_data["watches"] if w["id"] != id]

    if len(watch_data["watches"]) == original_count:
        typer.secho(f"Watch ID '{id}' not found", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Save
    watch_path.write_text(json.dumps(watch_data, indent=2), encoding='utf-8')
    typer.secho(f"Removed watch '{id}'", fg=typer.colors.GREEN)


@watch_app.command("history")
def watch_history_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    since: str | None = typer.Option(None, "--since", help="Filter by time (e.g., 7d, 1h)"),
    id: str | None = typer.Option(None, "--id", help="Filter by watch ID"),
    limit: int = typer.Option(50, "--limit", help="Maximum entries to show"),
    format: str = typer.Option("table", "--format", help="Output format: table, json"),
) -> None:
    """View watch history."""
    from datetime import datetime, timedelta
    from .context import load_context
    from .watch.history import read_watch_history, format_history_table, format_history_json

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Parse since filter
    since_ts = None
    if since:
        since_ts = parse_time_delta(since)

    # Read history
    entries = read_watch_history(
        ctx,
        since=since_ts,
        watch_id=id,
        limit=limit,
    )

    # Format output
    if format == "json":
        typer.echo(format_history_json(entries))
    else:
        typer.echo(format_history_table(entries))


def parse_time_delta(s: str) -> datetime:
    """Parse time delta string like '7d', '1h' into datetime."""
    from datetime import datetime, timedelta

    if s.endswith('d'):
        days = int(s[:-1])
        return datetime.utcnow() - timedelta(days=days)
    elif s.endswith('h'):
        hours = int(s[:-1])
        return datetime.utcnow() - timedelta(hours=hours)
    elif s.endswith('m'):
        minutes = int(s[:-1])
        return datetime.utcnow() - timedelta(minutes=minutes)
    else:
        raise ValueError(f"Invalid time delta: {s}")


# Add gateway subcommand group
gateway_app = typer.Typer(help="Gateway server commands")
app.add_typer(gateway_app, name="gateway")


@gateway_app.command("serve")
def gateway_serve(
    host: str = typer.Option(None, help="Host to bind (overrides config)"),
    port: int = typer.Option(None, help="Port to bind (overrides config)"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev only)")
):
    """
    Start the gateway server.

    Example:
        chinvex gateway serve --port 7778
    """
    import sys
    import uvicorn
    from .gateway.config import load_gateway_config

    config = load_gateway_config()

    # Check token is configured
    if not config.token:
        typer.echo(f"Error: {config.token_env} environment variable not set", err=True)
        typer.echo("Run 'chinvex gateway token-generate' to create a token", err=True)
        sys.exit(1)

    final_host = host or config.host
    final_port = port or config.port

    typer.echo(f"Starting Chinvex Gateway on {final_host}:{final_port}")
    typer.echo(f"Context allowlist: {config.context_allowlist or 'all contexts'}")
    typer.echo(f"Server-side LLM: {'enabled' if config.enable_server_llm else 'disabled'}")

    uvicorn.run(
        "chinvex.gateway.app:app",
        host=final_host,
        port=final_port,
        reload=reload
    )


@gateway_app.command("token-generate")
def gateway_token_generate():
    """
    Generate a new API token.

    Example:
        chinvex gateway token-generate
    """
    import secrets

    new_token = secrets.token_urlsafe(32)

    typer.echo("Generated new API token:")
    typer.echo()
    typer.echo(f"export CHINVEX_API_TOKEN={new_token}")
    typer.echo()
    typer.echo("Add this to your environment or secrets manager.")
    typer.echo("For ChatGPT Actions, use this token in the API Key field.")
    typer.echo()
    typer.echo("If using start_gateway.ps1, update the token in that file.")


@gateway_app.command("token-rotate")
def gateway_token_rotate():
    """
    Rotate API token (generates new, shows old).

    Example:
        chinvex gateway token-rotate
    """
    import secrets
    from .gateway.config import load_gateway_config

    config = load_gateway_config()
    old_token = config.token
    new_token = secrets.token_urlsafe(32)

    typer.echo("Token rotation:")
    typer.echo()
    if old_token:
        typer.echo(f"Old token: {old_token[:8]}...{old_token[-8:]}")
    else:
        typer.echo("Old token: (none)")
    typer.echo(f"New token: {new_token}")
    typer.echo()
    typer.echo(f"export CHINVEX_API_TOKEN={new_token}")
    typer.echo()
    typer.echo("Update this in:")
    typer.echo("- Environment variables")
    typer.echo("- ChatGPT Actions configuration")
    typer.echo("- start_gateway.ps1 (if using that script)")


@gateway_app.command("status")
def gateway_status():
    """
    Check gateway status and configuration.

    Example:
        chinvex gateway status
    """
    from .gateway.config import load_gateway_config
    from .context import list_contexts

    config = load_gateway_config()
    contexts = list_contexts()

    typer.echo("Gateway Configuration:")
    typer.echo(f"  Host: {config.host}")
    typer.echo(f"  Port: {config.port}")
    typer.echo(f"  Token configured: {'Yes' if config.token else 'No'}")
    typer.echo(f"  Server-side LLM: {'Enabled' if config.enable_server_llm else 'Disabled'}")
    typer.echo()
    typer.echo("Contexts:")
    if config.context_allowlist:
        typer.echo(f"  Allowlist: {', '.join(config.context_allowlist)}")
    else:
        typer.echo(f"  All contexts available ({len(contexts)} total)")
    typer.echo()
    typer.echo("Rate Limits:")
    typer.echo(f"  Per minute: {config.rate_limit.requests_per_minute}")
    typer.echo(f"  Per hour: {config.rate_limit.requests_per_hour}")
    typer.echo()
    typer.echo(f"Audit log: {config.audit_log_path}")


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
    from chinvex.context import load_context
    from chinvex.archive import archive_old_documents
    from chinvex.storage import Storage

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)
    db_path = ctx.index.sqlite_path
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
            days = 180  # Default

        # Run archive
        count = archive_old_documents(storage, age_threshold_days=days, dry_run=not force)

        if force:
            print(f"Archived {count} docs (older than {days}d)")
        else:
            print(f"Would archive {count} docs (dry-run)")
            print("Use --force to execute")

    elif action == "list":
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
                title = (doc["title"] or "")[:39]
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

    else:
        print(f"Unknown action: {action}")
        raise typer.Exit(1)


@digest_app.command("generate")
def digest_generate_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    since: str = typer.Option("24h", "--since", help="Time window (e.g., 24h, 7d)"),
    date: str | None = typer.Option(None, "--date", help="Generate for specific date (YYYY-MM-DD)"),
    push: str | None = typer.Option(None, "--push", help="Push notification (e.g., ntfy)"),
) -> None:
    """Generate digest for a context."""
    from .context import load_context
    from .context_cli import get_contexts_root
    from .digest import generate_digest
    from pathlib import Path
    import re

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Parse since
    match = re.match(r"(\d+)(h|d)", since)
    if not match:
        typer.secho(f"Invalid --since format: {since}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    amount, unit = match.groups()
    since_hours = int(amount) if unit == "h" else int(amount) * 24

    # Determine output date
    if date:
        from datetime import datetime
        output_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    else:
        from datetime import datetime
        output_date = datetime.now().strftime("%Y-%m-%d")

    # Setup paths
    context_dir = contexts_root / context
    ingest_runs_log = context_dir / "ingest_runs.jsonl"
    watch_history_log = context_dir / "watch_history.jsonl"

    # Find STATE.md (walk up from context to find repo root)
    state_md = None
    search_dir = Path.cwd()
    for _ in range(5):  # Max 5 levels up
        candidate = search_dir / "docs" / "memory" / "STATE.md"
        if candidate.exists():
            state_md = candidate
            break
        search_dir = search_dir.parent

    # Output paths
    digests_dir = context_dir / "digests"
    digests_dir.mkdir(parents=True, exist_ok=True)
    output_md = digests_dir / f"{output_date}.md"
    output_json = digests_dir / f"{output_date}.json"

    generate_digest(
        context_name=context,
        ingest_runs_log=ingest_runs_log if ingest_runs_log.exists() else None,
        watch_history_log=watch_history_log if watch_history_log.exists() else None,
        state_md=state_md,
        output_md=output_md,
        output_json=output_json,
        since_hours=since_hours
    )

    typer.secho(f"Digest generated: {output_md}", fg=typer.colors.GREEN)

    # Push notification if requested
    if push == "ntfy":
        _push_ntfy_notification(context, output_md)


def _push_ntfy_notification(context: str, digest_path: Path) -> None:
    """Push notification to ntfy."""
    import os
    import requests

    topic = os.getenv("CHINVEX_NTFY_TOPIC")
    server = os.getenv("CHINVEX_NTFY_SERVER", "https://ntfy.sh")

    if not topic:
        typer.secho("Warning: CHINVEX_NTFY_TOPIC not set, skipping notification", fg=typer.colors.YELLOW)
        return

    # Read digest stats
    content = digest_path.read_text()
    # Extract simple summary
    message = f"Chinvex digest ready for {context}"

    try:
        response = requests.post(
            f"{server}/{topic}",
            data=message,
            headers={"Title": f"Chinvex Digest - {context}"}
        )
        response.raise_for_status()
        typer.secho("Notification sent to ntfy", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to send notification: {e}", fg=typer.colors.RED)


@app.command("brief")
def brief_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    output: Path | None = typer.Option(None, "--output", help="Output file (default: stdout)"),
    repo_root: Path | None = typer.Option(None, "--repo-root", help="Repository root (auto-detect if not provided)"),
) -> None:
    """Generate session brief for a context."""
    from .context import load_context
    from .brief import generate_brief

    contexts_root = get_contexts_root()
    ctx = load_context(context, contexts_root)

    # Auto-detect repo root
    if not repo_root:
        repo_root = _find_repo_root(Path.cwd())

    # Setup paths
    state_md = repo_root / "docs" / "memory" / "STATE.md" if repo_root else None
    constraints_md = repo_root / "docs" / "memory" / "CONSTRAINTS.md" if repo_root else None
    decisions_md = repo_root / "docs" / "memory" / "DECISIONS.md" if repo_root else None

    context_dir = contexts_root / context
    digests_dir = context_dir / "digests"
    latest_digest = _find_latest_digest(digests_dir)
    watch_history_log = context_dir / "watch_history.jsonl"

    # Generate brief
    if output:
        output_path = output
    else:
        import tempfile
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        output_path = Path(temp.name)
        temp.close()

    generate_brief(
        context_name=context,
        state_md=state_md,
        constraints_md=constraints_md,
        decisions_md=decisions_md,
        latest_digest=latest_digest,
        watch_history_log=watch_history_log,
        output=output_path
    )

    if output:
        typer.secho(f"Brief generated: {output_path}", fg=typer.colors.GREEN)
    else:
        # Print to stdout
        content = output_path.read_text()
        typer.echo(content)
        output_path.unlink()  # Clean up temp file


def _find_repo_root(start_dir: Path) -> Path | None:
    """Walk up to find repo root (has .git or docs/memory)."""
    current = start_dir
    for _ in range(5):  # Max 5 levels
        if (current / ".git").exists() or (current / "docs" / "memory").exists():
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    return None


def _find_latest_digest(digests_dir: Path) -> Path | None:
    """Find most recent digest by filename (YYYY-MM-DD.md)."""
    if not digests_dir.exists():
        return None

    digests = sorted(digests_dir.glob("*.md"), reverse=True)
    return digests[0] if digests else None


@sync_app.command("start")
def sync_start():
    """Start the sync daemon"""
    from .sync.cli import sync_start_cmd
    sync_start_cmd()


@sync_app.command("stop")
def sync_stop():
    """Stop the sync daemon"""
    from .sync.cli import sync_stop_cmd
    sync_stop_cmd()


@sync_app.command("status")
def sync_status():
    """Show sync daemon status"""
    from .sync.cli import sync_status_cmd
    sync_status_cmd()


@sync_app.command("ensure-running")
def sync_ensure_running():
    """Start daemon if not running (idempotent)"""
    from .sync.cli import sync_ensure_running_cmd
    sync_ensure_running_cmd()


@sync_app.command("reconcile-sources")
def sync_reconcile_sources():
    """Update watcher sources from contexts (restarts watcher)"""
    from .sync.cli import sync_reconcile_sources_cmd
    sync_reconcile_sources_cmd()


@hook_app.command("install")
def hook_install(context: str = typer.Option(None, help="Context name (inferred from folder if omitted)")):
    """Install post-commit hook in current git repository"""
    from .hooks.cli import hook_install_cmd
    hook_install_cmd(context)


@hook_app.command("uninstall")
def hook_uninstall():
    """Uninstall post-commit hook from current repository"""
    from .hooks.cli import hook_uninstall_cmd
    hook_uninstall_cmd()


@hook_app.command("status")
def hook_status():
    """Show git hook installation status"""
    from .hooks.cli import hook_status_cmd
    hook_status_cmd()


@app.command()
def archive(
    context: str = typer.Option(..., help="Context name"),
    apply_constraints: bool = typer.Option(False, help="Apply age and count constraints"),
    age_days: int = typer.Option(None, help="Archive chunks older than N days"),
    max_chunks: int = typer.Option(None, help="Keep only N most recent chunks"),
    quiet: bool = typer.Option(False, help="Suppress output"),
):
    """
    Archive old chunks to reduce active index size.

    Use --apply-constraints to read from context.json constraints config.
    Or specify --age-days and/or --max-chunks explicitly.
    """
    from .context import load_context
    from .context_cli import get_contexts_root
    from .archive import archive_by_age, archive_by_count
    from .storage import Storage

    contexts_root = get_contexts_root()
    ctx_config = load_context(context, contexts_root)

    db_path = ctx_config.index.sqlite_path
    storage = Storage(db_path)

    # Determine constraints
    if apply_constraints:
        # Read from context config
        constraints = getattr(ctx_config, "constraints", None)
        if constraints:
            age_days = constraints.get("archive_after_days", 90)
            max_chunks = constraints.get("max_chunks", 10000)
        else:
            if not quiet:
                typer.echo("No constraints defined in context config")
            storage.close()
            return

    # Apply age constraint
    if age_days:
        stats = archive_by_age(storage, age_days)
        if not quiet:
            typer.echo(f"Archived {stats.archived_count} chunks older than {age_days} days")

    # Apply count constraint
    if max_chunks:
        stats = archive_by_count(storage, max_chunks)
        if not quiet:
            typer.echo(f"Archived {stats.archived_count} chunks to stay under {max_chunks}")

    storage.close()

    if not quiet:
        typer.secho("Archive complete", fg=typer.colors.GREEN)


@app.command()
def status(
    contexts_root: Path = typer.Option(
        None,
        help="Root directory for contexts (default: CHINVEX_CONTEXTS_ROOT env or P:/ai_memory/contexts)"
    ),
    regenerate: bool = typer.Option(False, help="Regenerate from STATUS.json files")
):
    """Show global system status."""
    import os
    from .cli_status import read_global_status, generate_status_from_contexts

    # Resolve contexts root
    if contexts_root is None:
        contexts_root = Path(os.getenv("CHINVEX_CONTEXTS_ROOT", "P:/ai_memory/contexts"))

    if regenerate:
        output = generate_status_from_contexts(contexts_root)
        # Write back to GLOBAL_STATUS.md
        global_status_md = contexts_root / "GLOBAL_STATUS.md"
        global_status_md.write_text(output, encoding="utf-8")
    else:
        output = read_global_status(contexts_root)

    print(output)


# ================================
# Bootstrap Commands
# ================================

@bootstrap_app.command()
def install(
    contexts_root: Path = typer.Option(
        Path("P:/ai_memory/contexts"),
        help="Root directory for contexts"
    ),
    indexes_root: Path = typer.Option(
        Path("P:/ai_memory/indexes"),
        help="Root directory for indexes"
    ),
    ntfy_topic: str = typer.Option(..., prompt=True, help="ntfy.sh topic for notifications"),
    morning_brief_time: str = typer.Option("07:00", help="Morning brief time (HH:MM)"),
    profile_path: Path = typer.Option(
        Path(os.path.expandvars(r"%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1")),
        help="PowerShell profile path"
    )
):
    """Install Chinvex bootstrap components."""
    from .bootstrap.cli import bootstrap_install
    bootstrap_install(contexts_root, indexes_root, ntfy_topic, profile_path, morning_brief_time)


@bootstrap_app.command()
def status():
    """Show bootstrap installation status."""
    from .bootstrap.cli import bootstrap_status

    status = bootstrap_status()

    print("Chinvex Bootstrap Status:")
    print(f"  Watcher: {'[OK] Running' if status['watcher_running'] else '[X] Stopped'}")
    print(f"  Sweep Task: {'[OK] Installed' if status['sweep_task_installed'] else '[X] Not installed'}")
    print(f"  Brief Task: {'[OK] Installed' if status['brief_task_installed'] else '[X] Not installed'}")
    print(f"  Env Vars: {'[OK] Set' if status['env_vars_set'] else '[X] Not set'}")


@bootstrap_app.command()
def uninstall(
    profile_path: Path = typer.Option(
        Path(os.path.expandvars(r"%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1")),
        help="PowerShell profile path"
    )
):
    """Uninstall Chinvex bootstrap components."""
    confirm = typer.confirm("This will remove scheduled tasks, env vars, and profile changes. Continue?")
    if not confirm:
        print("Uninstall cancelled.")
        return

    from .bootstrap.cli import bootstrap_uninstall
    bootstrap_uninstall(profile_path)


@app.command("update-memory")
def update_memory_cmd(
    context: str = typer.Option(..., "--context", "-c", help="Context name"),
    commit: bool = typer.Option(False, "--commit", help="Auto-commit changes (default: review mode)")
):
    """Update memory files (STATE.md, CONSTRAINTS.md, DECISIONS.md) from git history.

    Review mode (default): Shows diff without committing.
    Commit mode (--commit): Auto-commits with 'docs: update memory files'.
    """
    import json
    import subprocess
    from chinvex.memory_orchestrator import update_memory_files, get_memory_diff

    # Load context config to get repo paths
    contexts_root = get_contexts_root()
    ctx_dir = contexts_root / context
    config_path = ctx_dir / "context.json"

    if not config_path.exists():
        typer.echo(f"Error: Context '{context}' not found", err=True)
        raise typer.Exit(1)

    config = json.loads(config_path.read_text())

    repos = config.get("includes", {}).get("repos", [])
    if not repos:
        typer.echo(f"Error: No repos configured for context '{context}'", err=True)
        raise typer.Exit(1)

    # Use first repo (multi-repo support is future work)
    repo_root = Path(repos[0]["path"]) if isinstance(repos[0], dict) else Path(repos[0])
    if not repo_root.exists():
        typer.echo(f"Error: Repo not found: {repo_root}", err=True)
        raise typer.Exit(1)

    memory_dir = repo_root / "docs" / "memory"
    state_file = memory_dir / "STATE.md"
    constraints_file = memory_dir / "CONSTRAINTS.md"
    decisions_file = memory_dir / "DECISIONS.md"

    # Read old content for diff
    old_state = state_file.read_text() if state_file.exists() else ""
    old_constraints = constraints_file.read_text() if constraints_file.exists() else ""
    old_decisions = decisions_file.read_text() if decisions_file.exists() else ""

    # Run update
    result = update_memory_files(repo_root)

    if result.commits_processed == 0:
        typer.echo("No new commits since last update. Memory files are up to date.")
        return

    # Read new content
    new_state = state_file.read_text() if state_file.exists() else ""
    new_constraints = constraints_file.read_text() if constraints_file.exists() else ""
    new_decisions = decisions_file.read_text() if decisions_file.exists() else ""

    # Show summary
    typer.echo(f"Processed {result.commits_processed} commits")
    typer.echo(f"Analyzed {result.files_analyzed} spec/plan files")
    typer.echo(f"Updated {result.files_changed} memory files")

    if result.bounded_inputs_triggered:
        typer.echo("WARNING: Bounded inputs limit reached - some commits/files skipped", err=True)

    # Show diffs in review mode
    if not commit:
        typer.echo("\n=== CHANGES (review mode - not committed) ===\n")

        if new_state != old_state:
            typer.echo(get_memory_diff("STATE.md", old_state, new_state))

        if new_constraints != old_constraints:
            typer.echo(get_memory_diff("CONSTRAINTS.md", old_constraints, new_constraints))

        if new_decisions != old_decisions:
            typer.echo(get_memory_diff("DECISIONS.md", old_decisions, new_decisions))

        typer.echo("\nRun with --commit to auto-commit these changes.")
    else:
        # Commit mode
        if result.files_changed > 0:
            subprocess.run(["git", "add", "docs/memory/"], cwd=repo_root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "docs: update memory files"],
                cwd=repo_root,
                check=True
            )
            typer.echo("Changes committed.")
        else:
            typer.echo("No changes to commit.")


@app.command()
def eval(
    context: str = typer.Option(..., "--context", "-c", help="Context name to evaluate"),
    k: int | None = typer.Option(None, "--k", help="Override K value for all queries")
):
    """Run evaluation suite against golden queries.

    Loads golden queries for the specified context and runs retrieval evaluation.
    Reports hit rate, MRR, and latency metrics.
    Compares to baseline and fails if performance regresses.
    """
    from .eval_runner import run_evaluation
    from .eval_baseline import load_baseline_metrics, compare_to_baseline

    try:
        # Run evaluation
        typer.echo(f"Running evaluation for context: {context}")
        metrics = run_evaluation(context_name=context, k=k)

        # Display results
        typer.echo(f"\nEvaluation Results:")
        typer.echo(f"  Hit Rate@5: {metrics['hit_rate'] * 100:.1f}%")
        typer.echo(f"  MRR: {metrics['mrr']:.3f}")
        typer.echo(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
        typer.echo(f"  Passed: {metrics['passed']}/{metrics['total']}")
        typer.echo(f"  Failed: {metrics['failed']}/{metrics['total']}")

        # Compare to baseline
        try:
            from pathlib import Path
            baseline_file = Path("tests/eval/baseline_metrics.json")
            baseline = load_baseline_metrics(baseline_file, context)

            if baseline:
                current_metrics = type('EvalMetrics', (), {
                    'hit_rate': metrics['hit_rate'],
                    'mrr': metrics['mrr'],
                    'avg_latency_ms': metrics['avg_latency_ms']
                })()

                comparison = compare_to_baseline(current_metrics, baseline)

                typer.echo(f"\nBaseline Comparison:")
                typer.echo(f"  Baseline Hit Rate: {baseline.hit_rate * 100:.1f}%")
                typer.echo(f"  Change: {comparison.hit_rate_change * 100:+.1f}%")

                if comparison.passed:
                    typer.echo("  Status: PASS")
                else:
                    typer.echo(f"  Status: REGRESSION (below baseline threshold)")
                    raise typer.Exit(1)

        except FileNotFoundError:
            typer.echo("\nNo baseline metrics found. Run with --save-baseline to create.")

    except Exception as e:
        typer.echo(f"Error running evaluation: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
