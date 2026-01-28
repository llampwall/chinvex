# src/chinvex/scoring.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


FTS_WEIGHT = 0.6
VEC_WEIGHT = 0.4


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalize within the candidate set for this query.
    Returns values in [0, 1].
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)  # all equal = all max
    return [(s - min_s) / (max_s - min_s) for s in scores]


def blend_scores(fts_norm: float | None, vec_norm: float | None) -> float:
    """
    Combine NORMALIZED FTS and vector scores with weight renormalization.

    If only one score present, use 100% of that signal.
    If both present, blend with FTS_WEIGHT and VEC_WEIGHT.
    """
    if fts_norm is not None and vec_norm is not None:
        return fts_norm * FTS_WEIGHT + vec_norm * VEC_WEIGHT
    elif fts_norm is not None:
        return fts_norm  # 100% of available signal
    elif vec_norm is not None:
        return vec_norm
    else:
        return 0.0


def rank_score(blended: float, source_type: str, weights: dict[str, float]) -> float:
    """
    Apply source-type weight as a post-retrieval prior.
    """
    weight = weights.get(source_type, 0.5)  # default if unknown
    return blended * weight


@dataclass
class RankedResult:
    """Result from merge_and_rank with all score components."""
    chunk_id: str
    text: str
    source_uri: str
    source_type: str
    fts_score: float | None
    vector_score: float | None
    blended_score: float
    rank_score: float
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    updated_at: str | None = None
    meta: dict | None = None


def merge_and_rank(fts_results, vec_results, storage, weights, k=8):
    """
    Merge FTS and vector search results with normalized scoring.

    Args:
        fts_results: FTS search results (list of dicts with chunk_id, rank)
        vec_results: Vector search results (dict with ids, distances)
        storage: Storage instance for fetching chunk data
        weights: Source-type weights dict
        k: Number of results to return

    Returns:
        List of RankedResult objects
    """
    import json

    # Build candidate set
    candidates: dict[str, dict] = {}

    # Add FTS results
    for row in fts_results:
        chunk_id = row["chunk_id"]
        # sqlite3.Row doesn't have .get(), check with 'in' operator
        rank = row["rank"] if "rank" in row.keys() else 0
        candidates[chunk_id] = {
            "row": row,
            "fts_raw": float(rank),
            "vec_raw": None,
        }

    # Add vector results
    vec_ids = vec_results.get("ids", [[]])[0]
    vec_distances = vec_results.get("distances", [[]])[0]
    for chunk_id, dist in zip(vec_ids, vec_distances):
        if dist is None:
            continue
        if chunk_id not in candidates:
            # Fetch row from storage
            cur = storage.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cur.fetchone()
            if row is None:
                continue
            candidates[chunk_id] = {
                "row": row,
                "fts_raw": None,
                "vec_raw": float(dist),
            }
        else:
            candidates[chunk_id]["vec_raw"] = float(dist)

    # Normalize FTS scores (BM25 ranks - lower is better, invert)
    fts_raws = [c["fts_raw"] for c in candidates.values() if c["fts_raw"] is not None]
    if fts_raws:
        fts_normalized = normalize_scores([-r for r in fts_raws])  # invert ranks
        fts_map = dict(zip([cid for cid, c in candidates.items() if c["fts_raw"] is not None], fts_normalized))
    else:
        fts_map = {}

    # Normalize vector scores (cosine distance - lower is better, invert)
    vec_raws = [c["vec_raw"] for c in candidates.values() if c["vec_raw"] is not None]
    if vec_raws:
        vec_normalized = normalize_scores([1.0 / (1.0 + d) for d in vec_raws])
        vec_map = dict(zip([cid for cid, c in candidates.items() if c["vec_raw"] is not None], vec_normalized))
    else:
        vec_map = {}

    # Blend scores and create results
    results: list[RankedResult] = []
    for chunk_id, data in candidates.items():
        row = data["row"]
        fts_norm = fts_map.get(chunk_id)
        vec_norm = vec_map.get(chunk_id)

        blended = blend_scores(fts_norm, vec_norm)

        # Apply source-type weight
        source_type = row["source_type"]
        final_score = rank_score(blended, source_type, weights)

        # Parse metadata
        meta_json = row["meta_json"] if "meta_json" in row.keys() and row["meta_json"] else None
        meta = json.loads(meta_json) if meta_json else {}

        # Build source URI
        if source_type == "repo":
            source_uri = meta.get("path", row["doc_id"])
        else:
            source_uri = row["doc_id"]

        # Get updated_at if available
        updated_at = row["updated_at"] if "updated_at" in row.keys() else None

        results.append(RankedResult(
            chunk_id=chunk_id,
            text=row["text"],
            source_uri=source_uri,
            source_type=source_type,
            fts_score=fts_norm,
            vector_score=vec_norm,
            blended_score=blended,
            rank_score=final_score,
            line_start=meta.get("line_start"),
            line_end=meta.get("line_end"),
            char_start=meta.get("char_start"),
            char_end=meta.get("char_end"),
            updated_at=updated_at,
            meta=meta
        ))

    results.sort(key=lambda r: r.rank_score, reverse=True)
    return results[:k]
