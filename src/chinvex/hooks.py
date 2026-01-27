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

        recently_changed = extract_recently_changed(
            context=context.name,
            since=since,
            limit=20,
            db_path=getattr(context, 'db_path', None)
        )

        active_threads = extract_active_threads(
            context=context.name,
            days=7,
            limit=20,
            db_path=getattr(context, 'db_path', None)
        )

        # Extract TODOs from recently changed files
        todos = []
        # TODO: implement TODO extraction from changed files

        # Run watches (P1.4)
        watch_hits = []
        # TODO: load watches and run them

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
        # Production path
        state_dir = Path(f"P:/ai_memory/contexts/{context.name}")

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
