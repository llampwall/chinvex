# tests/test_scoring.py
from chinvex.scoring import normalize_scores, blend_scores, rank_score


def test_normalize_scores_minmax() -> None:
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = normalize_scores(scores)

    assert normalized[0] == 0.0  # min
    assert normalized[-1] == 1.0  # max
    assert 0.0 <= normalized[2] <= 1.0  # middle


def test_normalize_scores_empty() -> None:
    assert normalize_scores([]) == []


def test_normalize_scores_all_equal() -> None:
    scores = [5.0, 5.0, 5.0]
    normalized = normalize_scores(scores)

    assert all(s == 1.0 for s in normalized)


def test_blend_scores_both_present() -> None:
    fts_norm = 0.8
    vec_norm = 0.6

    blended = blend_scores(fts_norm, vec_norm)

    # FTS_WEIGHT=0.6, VEC_WEIGHT=0.4
    expected = (0.8 * 0.6) + (0.6 * 0.4)
    assert abs(blended - expected) < 0.001


def test_blend_scores_only_fts() -> None:
    blended = blend_scores(0.8, None)
    assert blended == 0.8


def test_blend_scores_only_vector() -> None:
    blended = blend_scores(None, 0.7)
    assert blended == 0.7


def test_blend_scores_neither() -> None:
    blended = blend_scores(None, None)
    assert blended == 0.0


def test_rank_score_applies_weight() -> None:
    blended = 0.8
    weight = 0.9  # codex_session weight

    rank = rank_score(blended, "codex_session", {"codex_session": 0.9})

    assert rank == 0.8 * 0.9


def test_rank_score_uses_default_for_unknown_type() -> None:
    blended = 0.8
    weights = {"repo": 1.0}

    rank = rank_score(blended, "unknown_type", weights)

    assert rank == 0.8 * 0.5  # default weight
