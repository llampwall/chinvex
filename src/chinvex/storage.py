from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from .util import now_iso

SCHEMA_VERSION = 1

_CONN: sqlite3.Connection | None = None
_CONN_PATH: Path | None = None


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = self._get_connection(db_path)

    def close(self) -> None:
        # Single shared connection per process; no-op on close.
        return None

    @classmethod
    def _get_connection(cls, db_path: Path) -> sqlite3.Connection:
        global _CONN, _CONN_PATH
        if _CONN is None or _CONN_PATH != db_path:
            if _CONN is not None:
                _CONN.close()
            _CONN = cls._connect(db_path)
            _CONN_PATH = db_path
        return _CONN

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def ensure_schema(self) -> None:
        self._check_fts5()

        # Create meta table first
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS meta(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )

        # Check schema version
        cur = self._execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        if row is None:
            # First time setup
            self._execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),)
            )
        else:
            stored_version = int(row["value"])
            if stored_version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"schema version mismatch: database has version {stored_version}, "
                    f"but code expects version {SCHEMA_VERSION}. "
                    f"Delete index folder or run migration script."
                )

        # Continue with existing table creation
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS documents(
              doc_id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              project TEXT,
              repo TEXT,
              title TEXT,
              updated_at TEXT,
              content_hash TEXT,
              meta_json TEXT
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
              chunk_id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              source_type TEXT NOT NULL,
              project TEXT,
              repo TEXT,
              ordinal INTEGER NOT NULL,
              text TEXT NOT NULL,
              updated_at TEXT,
              meta_json TEXT
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs(
              run_id TEXT PRIMARY KEY,
              started_at TEXT,
              finished_at TEXT,
              stats_json TEXT
            )
            """
        )
        self._execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, content='', tokenize='unicode61')
            """
        )
        # After chunks_fts creation
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS source_fingerprints (
                source_uri TEXT NOT NULL,
                context_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                size_bytes INTEGER,
                mtime_unix INTEGER,
                content_sha256 TEXT,
                thread_updated_at TEXT,
                last_turn_id TEXT,
                parser_version TEXT NOT NULL,
                chunker_version TEXT NOT NULL,
                embedded_model TEXT,
                last_ingested_at_unix INTEGER NOT NULL,
                last_status TEXT NOT NULL,
                last_error TEXT,
                PRIMARY KEY (source_uri, context_name)
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprints_type ON source_fingerprints(source_type)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprints_status ON source_fingerprints(last_status)"
        )
        self.conn.commit()

    def _check_fts5(self) -> None:
        try:
            self._execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(text)")
            self._execute("DROP TABLE IF EXISTS _fts5_test")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "FTS5 not available in this Python/SQLite build. "
                "Install a Python build with SQLite FTS5 enabled."
            ) from exc

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        cur = self._execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.fetchone()

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[sqlite3.Row]:
        """Fetch multiple chunks by their IDs with source_uri from documents."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        cur = self._execute(
            f"""
            SELECT c.*, d.source_uri
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            chunk_ids
        )
        return cur.fetchall()

    def upsert_document(
        self,
        *,
        doc_id: str,
        source_type: str,
        source_uri: str,
        project: str | None,
        repo: str | None,
        title: str | None,
        updated_at: str,
        content_hash: str,
        meta_json: str,
    ) -> None:
        self._execute(
            """
            INSERT INTO documents(doc_id, source_type, source_uri, project, repo, title, updated_at, content_hash, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
              source_type=excluded.source_type,
              source_uri=excluded.source_uri,
              project=excluded.project,
              repo=excluded.repo,
              title=excluded.title,
              updated_at=excluded.updated_at,
              content_hash=excluded.content_hash,
              meta_json=excluded.meta_json
            """,
            (doc_id, source_type, source_uri, project, repo, title, updated_at, content_hash, meta_json),
        )
        self.conn.commit()

    def delete_chunks_for_doc(self, doc_id: str) -> list[str]:
        cur = self._execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        chunk_ids = [row["chunk_id"] for row in cur.fetchall()]
        self._execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        self._execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return chunk_ids

    def upsert_chunks(self, rows: Iterable[tuple]) -> None:
        self._executemany(
            """
            INSERT INTO chunks(chunk_id, doc_id, source_type, project, repo, ordinal, text, updated_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              doc_id=excluded.doc_id,
              source_type=excluded.source_type,
              project=excluded.project,
              repo=excluded.repo,
              ordinal=excluded.ordinal,
              text=excluded.text,
              updated_at=excluded.updated_at,
              meta_json=excluded.meta_json
            """,
            rows,
        )
        self.conn.commit()

    def upsert_fts(self, rows: Iterable[tuple]) -> None:
        self._executemany(
            "INSERT OR REPLACE INTO chunks_fts(rowid, text) VALUES ((SELECT rowid FROM chunks WHERE chunk_id = ?), ?)",
            rows,
        )
        self.conn.commit()

    def search_fts(self, query: str, limit: int = 30, filters: dict | None = None) -> list[sqlite3.Row]:
        filters = filters or {}
        clauses = []
        params: list = []
        if filters.get("source_type"):
            clauses.append("chunks.source_type = ?")
            params.append(filters["source_type"])
        if filters.get("project"):
            clauses.append("chunks.project = ?")
            params.append(filters["project"])
        if filters.get("repo"):
            clauses.append("chunks.repo = ?")
            params.append(filters["repo"])
        sql = f"""
            SELECT chunks.chunk_id, chunks.text, chunks.source_type, chunks.project, chunks.repo,
                   chunks.doc_id, chunks.ordinal, chunks.updated_at, chunks.meta_json,
                   bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks ON chunks_fts.rowid = chunks.rowid
            WHERE chunks_fts MATCH ?
            {('AND ' + ' AND '.join(clauses)) if clauses else ''}
            ORDER BY rank
            LIMIT ?
        """
        params = [query] + params + [limit]
        cur = self._execute(sql, params)
        return cur.fetchall()

    def record_run(self, run_id: str, started_at: str, stats_json: str) -> None:
        finished = now_iso()
        self._execute(
            """
            INSERT INTO ingestion_runs(run_id, started_at, finished_at, stats_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, started_at, finished, stats_json),
        )
        self.conn.commit()

    def _execute(self, sql: str, params: tuple | list | None = None) -> sqlite3.Cursor:
        try:
            if params is None:
                return self.conn.execute(sql)
            return self.conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if "disk I/O error" in str(exc):
                return self._retry_once(sql, params)
            raise

    def _executemany(self, sql: str, rows: Iterable[tuple]) -> None:
        try:
            self.conn.executemany(sql, rows)
        except sqlite3.OperationalError as exc:
            if "disk I/O error" in str(exc):
                self._retry_once(sql, rows, many=True)
            else:
                raise

    def _retry_once(self, sql: str, params: tuple | list | Iterable[tuple] | None, *, many: bool = False):
        global _CONN, _CONN_PATH
        try:
            self.conn.close()
        except Exception:
            pass
        _CONN = None
        _CONN_PATH = None
        time.sleep(random.uniform(0.25, 0.5))
        self.conn = self._connect(self.db_path)
        _CONN = self.conn
        _CONN_PATH = self.db_path
        try:
            if many:
                self.conn.executemany(sql, params)  # type: ignore[arg-type]
            else:
                if params is None:
                    return self.conn.execute(sql)
                return self.conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if "disk I/O error" in str(exc):
                raise RuntimeError(
                    "SQLite disk I/O error persisted after retry. Move index_dir to a local disk."
                ) from exc
            raise

    def upsert_fingerprint(
        self,
        *,
        source_uri: str,
        context_name: str,
        source_type: str,
        doc_id: str,
        parser_version: str,
        chunker_version: str,
        embedded_model: str | None,
        last_status: str,
        last_error: str | None,
        size_bytes: int | None = None,
        mtime_unix: int | None = None,
        content_sha256: str | None = None,
        thread_updated_at: str | None = None,
        last_turn_id: str | None = None,
    ) -> None:
        import time
        last_ingested_at_unix = int(time.time())
        self._execute(
            """
            INSERT INTO source_fingerprints(
                source_uri, context_name, source_type, doc_id,
                size_bytes, mtime_unix, content_sha256,
                thread_updated_at, last_turn_id,
                parser_version, chunker_version, embedded_model,
                last_ingested_at_unix, last_status, last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_uri, context_name) DO UPDATE SET
                source_type=excluded.source_type,
                doc_id=excluded.doc_id,
                size_bytes=excluded.size_bytes,
                mtime_unix=excluded.mtime_unix,
                content_sha256=excluded.content_sha256,
                thread_updated_at=excluded.thread_updated_at,
                last_turn_id=excluded.last_turn_id,
                parser_version=excluded.parser_version,
                chunker_version=excluded.chunker_version,
                embedded_model=excluded.embedded_model,
                last_ingested_at_unix=excluded.last_ingested_at_unix,
                last_status=excluded.last_status,
                last_error=excluded.last_error
            """,
            (
                source_uri, context_name, source_type, doc_id,
                size_bytes, mtime_unix, content_sha256,
                thread_updated_at, last_turn_id,
                parser_version, chunker_version, embedded_model,
                last_ingested_at_unix, last_status, last_error,
            ),
        )
        self.conn.commit()

    def get_fingerprint(self, source_uri: str, context_name: str) -> sqlite3.Row | None:
        cur = self._execute(
            "SELECT * FROM source_fingerprints WHERE source_uri = ? AND context_name = ?",
            (source_uri, context_name),
        )
        return cur.fetchone()
