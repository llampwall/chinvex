# src/chinvex/hooks.py
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from chinvex.state.models import StateJson
from chinvex.state.extractors import extract_recently_changed, extract_active_threads, extract_todos
from chinvex.state.renderer import render_state_md

log = logging.getLogger(__name__)


def post_ingest_hook(context, result):
    """
    Called after every ingest run to generate STATE.md.

    Args:
        context: Context object
        result: IngestRunResult from ingest

    Note:
        State generation failures DO NOT fail ingest (best-effort).
    """
    try:
        # Extract state components
        since = result.started_at

        db_path = str(context.index.sqlite_path) if hasattr(context, 'index') else None

        recently_changed = extract_recently_changed(
            context=context.name,
            since=since,
            limit=20,
            db_path=db_path
        )

        active_threads = extract_active_threads(
            context=context.name,
            days=7,
            limit=20,
            db_path=db_path
        )

        # Extract TODOs from recently changed files
        todos = []
        # TODO: implement TODO extraction from changed files

        # Run watches (P1.4)
        watch_hits = []
        if result.new_chunk_ids:
            from chinvex.watch.models import Watch
            from chinvex.watch.runner import run_watches, append_watch_history, create_watch_history_entry
            from chinvex.context_cli import get_contexts_root

            # Load watches from watch.json
            contexts_root = get_contexts_root()
            watch_path = contexts_root / context.name / "watch.json"

            if watch_path.exists():
                watch_data = json.loads(watch_path.read_text(encoding='utf-8'))
                watches = [
                    Watch(
                        id=w["id"],
                        query=w["query"],
                        min_score=w["min_score"],
                        enabled=w.get("enabled", True),
                        created_at=w.get("created_at", "")
                    )
                    for w in watch_data.get("watches", [])
                ]

                # Run watches against new chunks
                if watches:
                    hits = run_watches(context, result.new_chunk_ids, watches)

                    # Log hits to watch_history.jsonl
                    history_file = str(Path(context.index.sqlite_path).parent / "watch_history.jsonl")
                    for hit in hits:
                        entry = create_watch_history_entry(
                            watch_id=hit.watch_id,
                            query=hit.query,
                            hits=hit.hits,
                            run_id=result.run_id
                        )
                        append_watch_history(history_file, entry)

                    # Use WatchHit objects directly
                    watch_hits = hits

        # Create state
        state = StateJson(
            schema_version=1,
            context=context.name,
            generated_at=datetime.now(timezone.utc),
            last_ingest_run=result.run_id,
            generation_status="ok",
            generation_error=None,
            recently_changed=recently_changed,
            active_threads=active_threads,
            extracted_todos=todos,
            watch_hits=watch_hits,
            decisions=[],
            facts=[],
            annotations=[]
        )

    except Exception as e:
        log.error(f"State generation failed: {e}")
        # Create minimal error state
        state = StateJson(
            schema_version=1,
            context=context.name,
            generated_at=datetime.now(timezone.utc),
            last_ingest_run=result.run_id,
            generation_status="failed",
            generation_error=str(e),
            recently_changed=[],
            active_threads=[],
            extracted_todos=[],
            watch_hits=[],
            decisions=[],
            facts=[],
            annotations=[]
        )

    # Determine output directory
    if hasattr(context, 'state_dir'):
        # Test path override
        state_dir = Path(context.state_dir)
    else:
        # Use context root from environment or default
        from chinvex.context_cli import get_contexts_root
        contexts_root = get_contexts_root()
        state_dir = contexts_root / context.name

    state_dir.mkdir(parents=True, exist_ok=True)

    # Save state.json (following P0 pattern: dataclass.to_dict() -> json.dumps)
    state_path = state_dir / "state.json"
    state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding='utf-8')

    # Render and save STATE.md
    md = render_state_md(state)
    md_path = state_dir / "STATE.md"
    md_path.write_text(md, encoding='utf-8')

    if state.generation_status == "ok":
        log.info(f"STATE.md updated: {len(result.new_chunk_ids)} new chunks")
    else:
        log.warning(f"STATE.md updated with errors: {state.generation_error}")
