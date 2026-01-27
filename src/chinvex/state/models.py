# src/chinvex/state/models.py
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass(frozen=True)
class RecentlyChanged:
    """Document that changed recently."""
    doc_id: str
    source_type: str
    source_uri: str
    change_type: str  # "new" or "modified"
    changed_at: datetime
    summary: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d['changed_at'] = self.changed_at.isoformat()
        return d


@dataclass(frozen=True)
class ActiveThread:
    """Active Codex session thread."""
    id: str
    title: str
    status: str
    last_activity: datetime
    source: str

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d['last_activity'] = self.last_activity.isoformat()
        return d


@dataclass(frozen=True)
class ExtractedTodo:
    """TODO extracted from source code."""
    text: str
    source_uri: str
    line: int
    extracted_at: datetime

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d['extracted_at'] = self.extracted_at.isoformat()
        return d


@dataclass(frozen=True)
class WatchHit:
    """Watch query that matched new content."""
    watch_id: str
    query: str
    hits: list[dict]
    triggered_at: datetime

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d['triggered_at'] = self.triggered_at.isoformat()
        return d


@dataclass(frozen=True)
class StateJson:
    """State file schema (state.json)."""
    schema_version: int
    context: str
    generated_at: datetime
    last_ingest_run: str
    generation_status: str  # "ok", "partial", "failed"
    generation_error: str | None

    recently_changed: list[RecentlyChanged]
    active_threads: list[ActiveThread]
    extracted_todos: list[ExtractedTodo]
    watch_hits: list[WatchHit]
    decisions: list[dict] = field(default_factory=list)  # From LLM consolidator (P1.5)
    facts: list[dict] = field(default_factory=list)      # From LLM consolidator (P1.5)
    annotations: list[dict] = field(default_factory=list)  # Human annotations

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (following P0 pattern)."""
        return {
            'schema_version': self.schema_version,
            'context': self.context,
            'generated_at': self.generated_at.isoformat(),
            'last_ingest_run': self.last_ingest_run,
            'generation_status': self.generation_status,
            'generation_error': self.generation_error,
            'recently_changed': [item.to_dict() for item in self.recently_changed],
            'active_threads': [item.to_dict() for item in self.active_threads],
            'extracted_todos': [item.to_dict() for item in self.extracted_todos],
            'watch_hits': [item.to_dict() for item in self.watch_hits],
            'decisions': self.decisions,
            'facts': self.facts,
            'annotations': self.annotations
        }
