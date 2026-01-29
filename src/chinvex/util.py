from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    "out",
    ".venv",
    "venv",
    ".codex",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    ".eggs",
    "chroma",
    ".worktrees",
    ".claude",
    ".pnpm-store",
    ".pnpm",
    ".vscode",
    ".ruff_cache",
    ".cursor",
    ".qodo",
}

ALLOWED_EXTS = {".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_path(path: Path) -> str:
    return str(path.resolve())


def iso_from_mtime(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def read_text_utf8(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def should_exclude(path: Path, root: Path, excludes: list[str]) -> bool:
    """Check if path matches any exclude pattern relative to root."""
    rel_path = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in excludes)


def walk_files(root: Path, excludes: list[str] | None = None) -> Iterable[Path]:
    """Walk files in root, skipping SKIP_DIRS and paths matching exclude patterns."""
    excludes = excludes or []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in ALLOWED_EXTS and not should_exclude(path, root, excludes):
                yield path


def dump_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def in_venv() -> bool:
    return os.environ.get("VIRTUAL_ENV") is not None or getattr(os, "base_prefix", "") != getattr(os, "prefix", "")


def dataclass_to_json(obj: object) -> str:
    return dump_json(asdict(obj))


def normalize_path_for_dedup(path: str | Path) -> str:
    """
    Normalize path for deduplication:
    - Convert to absolute
    - Use forward slashes
    - Lowercase on Windows (case-insensitive)
    """
    abs_path = Path(path).resolve()
    normalized = abs_path.as_posix()

    if platform.system() == "Windows":
        normalized = normalized.lower()

    return normalized
