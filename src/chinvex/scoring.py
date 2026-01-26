# src/chinvex/scoring.py
from __future__ import annotations


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
