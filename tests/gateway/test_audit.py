import json
import tempfile
from pathlib import Path
from chinvex.gateway.audit import AuditLogger


def test_audit_logger_writes_jsonl():
    """Should write audit entries as JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log(
            request_id="req_123",
            endpoint="/v1/evidence",
            status=200,
            latency_ms=142,
            context="Chinvex",
            query="test query",
            client_ip="127.0.0.1"
        )

        lines = log_path.read_text().strip().split('\n')
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["request_id"] == "req_123"
        assert entry["endpoint"] == "/v1/evidence"
        assert entry["status"] == 200
        assert entry["latency_ms"] == 142
        assert entry["context"] == "Chinvex"
        assert entry["query_hash"].startswith("sha256:")
        assert entry["client_ip"] == "127.0.0.1"


def test_audit_logger_hashes_query():
    """Should hash query instead of logging raw text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log(
            request_id="req_123",
            endpoint="/v1/evidence",
            status=200,
            latency_ms=100,
            query="sensitive query text"
        )

        content = log_path.read_text()
        assert "sensitive query text" not in content
        assert "sha256:" in content


def test_audit_logger_creates_parent_dir():
    """Should create parent directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nested" / "dir" / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log(
            request_id="req_123",
            endpoint="/health",
            status=200,
            latency_ms=5
        )

        assert log_path.exists()
