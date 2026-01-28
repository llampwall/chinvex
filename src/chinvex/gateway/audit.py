"""Audit logging for gateway requests."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AuditEntry:
    """Audit log entry."""
    ts: str
    request_id: str
    endpoint: str
    context: Optional[str]
    query_hash: Optional[str]
    status: int
    latency_ms: int
    client_ip: Optional[str]


class AuditLogger:
    """Append-only audit logger."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        request_id: str,
        endpoint: str,
        status: int,
        latency_ms: int,
        context: Optional[str] = None,
        query: Optional[str] = None,
        client_ip: Optional[str] = None
    ):
        """
        Append audit entry to JSONL log.

        Args:
            request_id: Unique request identifier
            endpoint: Request path
            status: HTTP status code
            latency_ms: Response time in milliseconds
            context: Context name if applicable
            query: Query text (will be hashed, not stored raw)
            client_ip: Client IP address
        """
        entry = AuditEntry(
            ts=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            endpoint=endpoint,
            context=context,
            query_hash=self._hash_query(query) if query else None,
            status=status,
            latency_ms=latency_ms,
            client_ip=client_ip
        )

        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

    @staticmethod
    def _hash_query(query: str) -> str:
        """Hash query for privacy."""
        return "sha256:" + hashlib.sha256(query.encode()).hexdigest()[:16]
