# Connection Management in Chinvex

## Overview

Chinvex uses two database systems that require proper connection management:
1. **SQLite** (via `Storage` class) - for lexical search (FTS5)
2. **ChromaDB** (via `VectorStore` class) - for vector search

On Windows, improper connection handling can lead to file locks that prevent deletion and cause `PermissionError` issues.

## Implementation

### SQLite Connection Management

**Global Connection Pattern** (`storage.py`):
- Uses a module-level global connection (`_CONN`) shared across Storage instances
- Prevents "database is locked" errors from concurrent access
- Must be explicitly closed before file deletion

**Force Close Method**:
```python
Storage.force_close_global_connection()
```

Called in:
- `cli.py:185` - before single context purge
- `cli.py:1038` - before purging all contexts
- `ingest.py:458` - before clearing SQLite during re-ingestion
- `gateway/app.py:327` - during gateway shutdown

### ChromaDB Connection Management

**VectorStore Close Method** (`vectors.py`):
```python
vec_store.close()
```

This method:
1. Calls `client._system.stop()` to properly shut down ChromaDB's internal SQLite connections
2. Clears object references to allow garbage collection
3. Is idempotent (safe to call multiple times)

**Context Manager Support**:
```python
with VectorStore(chroma_dir) as vec_store:
    vec_store.count()
    # Automatically closed on exit
```

### Gateway Lifecycle Management

**Startup** (`gateway/app.py:247`):
- Creates test VectorStore and Storage instances for warmup
- **Now properly closes** the test VectorStore after warmup

**Shutdown** (`gateway/app.py:317`):
- Closes global SQLite connection via `Storage.force_close_global_connection()`
- VectorStore instances created per-request are garbage collected automatically

### Health Check Endpoints

**healthz.py**:
- Creates temporary VectorStore for readiness check
- **Now properly closes** it after the check to prevent connection buildup

## Why This Matters

### Windows File Locking
Windows holds file locks on SQLite databases until connections are explicitly closed. Without proper cleanup:
- `PermissionError` when deleting database files
- Gateway restarts fail due to lingering file handles
- Temp directory cleanup fails in tests

### Connection Pool Exhaustion
Long-running processes (gateway, daemon) accumulate connections if not cleaned up:
- Memory leaks
- Performance degradation
- Resource exhaustion

## Testing

See `tests/test_vector_store_cleanup.py` for connection cleanup verification:
- `test_vector_store_close()` - verifies close() releases file locks
- `test_vector_store_context_manager()` - verifies context manager cleanup
- `test_vector_store_close_idempotent()` - verifies safe multiple close()
- `test_vector_store_operations_after_close()` - verifies graceful failure after close

All tests pass on Windows, confirming ChromaDB SQLite connections are properly released.

## Best Practices

### Short-Lived Operations (CLI commands)
```python
# Python GC handles cleanup automatically
vectors = VectorStore(chroma_dir)
vectors.upsert(...)
# No explicit close needed
```

### Long-Lived Operations (gateway, daemon)
```python
# Explicit cleanup for warmup/health checks
vec_store = VectorStore(chroma_dir)
try:
    vec_store.count()
finally:
    vec_store.close()

# Or use context manager
with VectorStore(chroma_dir) as vec_store:
    vec_store.count()
```

### Before File Deletion
```python
# ALWAYS close connections before deleting files
Storage.force_close_global_connection()
vec_store.close()
shutil.rmtree(context_dir)
```

## Implementation Details

### ChromaDB Internals

ChromaDB's `PersistentClient` creates an internal `System` object (`client._system`) that manages:
- SQLite connection pool
- Background threads
- Resource handles

The `_system.stop()` method:
- Closes all SQLite connections
- Stops background threads
- Releases file handles

Simply clearing `client = None` is insufficient because:
- Python's GC is non-deterministic
- ChromaDB's finalizers may not run before file deletion
- Background threads may keep references alive

### Backward Compatibility

These changes are **fully backward compatible**:
- Existing code that doesn't call `close()` continues to work
- Python's GC eventually cleans up connections (just not immediately)
- The changes only add explicit cleanup for cases that need it

## Related Issues

- Gateway errors from lingering connections (see `logs/gateway-error-*.log`)
- Windows `PermissionError: [WinError 32]` when deleting database files
- Temp directory cleanup failures in test suite
