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
        stats = ingest_context(ctx, ollama_host_override=ollama_host)

        typer.secho(f"Ingestion complete for context '{context}':", fg=typer.colors.GREEN)
        typer.echo(f"  Documents: {stats['documents']}")
        typer.echo(f"  Chunks: {stats['chunks']}")
        typer.echo(f"  Skipped: {stats['skipped']}")
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
    config: Path = typer.Option(Path("config.json"), "--config", help="Path to config JSON"),
    k: int = typer.Option(8, "--k", help="Top K results"),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum score threshold"),
    source: str = typer.Option("all", "--source", help="all|repo|chat"),
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repo"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)
    if source not in {"all", "repo", "chat"}:
        raise typer.BadParameter("source must be one of: all, repo, chat")
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


@context_app.command("create")
def context_create_cmd(name: str = typer.Argument(..., help="Context name")) -> None:
    """Create a new context."""
    create_context(name)


@context_app.command("list")
def context_list_cmd() -> None:
    """List all contexts."""
    list_contexts_cli()


if __name__ == "__main__":
    app()
