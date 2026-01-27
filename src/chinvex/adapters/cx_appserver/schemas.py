# src/chinvex/adapters/cx_appserver/schemas.py
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class AppServerTurn(BaseModel):
    """
    Schema for a turn from app-server API.

    Discovered from actual /thread/resume responses.
    """
    turn_id: str
    ts: str  # ISO8601
    role: str  # user|assistant|tool|system
    text: str | None = None
    tool: dict | None = None
    attachments: list[dict] = Field(default_factory=list)
    meta: dict | None = None


class AppServerThread(BaseModel):
    """
    Schema for a thread from app-server API.

    Discovered from actual /thread/list and /thread/resume responses.
    """
    id: str
    title: str | None = None
    created_at: str  # ISO8601
    updated_at: str  # ISO8601
    turns: list[AppServerTurn] = Field(default_factory=list)
    links: dict = Field(default_factory=dict)


# Additional schemas for P1 implementation

class ThreadSummary(BaseModel):
    """Summary of a Codex thread from list endpoint."""
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int | None = None


class ThreadMessage(BaseModel):
    """Single message in a thread."""
    id: str
    role: str  # user|assistant|tool|system
    content: str | None
    timestamp: datetime
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None
    model: str | None = None


class ThreadDetail(BaseModel):
    """Full thread with all messages."""
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[ThreadMessage]
    workspace_id: str | None = None
    repo_paths: list[str] | None = None
