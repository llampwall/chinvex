from __future__ import annotations

from pathlib import Path

import typer

from .config import ConfigError, load_config
from .context_cli import create_context, get_contexts_root, list_contexts_cli
from .ingest import ingest
from .search import search
from .util import in_venv

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


def _load_config(config_path: Path):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("ingest")
def ingest_cmd(
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to ingest"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if not context and not config:
        typer.secho("Error: Must provide either --context or --config", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if context:
        # New context-based ingestion
        from .context import load_context
        from .ingest import ingest_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)
        result = ingest_context(ctx, ollama_host_override=ollama_host)

        typer.secho(f"Ingestion complete for context '{context}':", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {result.stats['documents']}")
        typer.echo(f"  Chunks: {result.stats['chunks']}")
        typer.echo(f"  Skipped: {result.stats['skipped']}")
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
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if source not in {"all", "repo", "chat", "codex_session"}:
        raise typer.BadParameter("source must be one of: all, repo, chat, codex_session")

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
def context_create_cmd(name: str = typer.Argument(..., help="Context name")) -> None:
    """Create a new context."""
    create_context(name)


@context_app.command("list")
def context_list_cmd() -> None:
    """List all contexts."""
    list_contexts_cli()


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


if __name__ == "__main__":
    app()
