import pytest

from autoresearch.research_harness.ranking_metrics import (
    RankingRow,
    ndcg_at_k,
    recall_at_k,
    summarize_ranking_metrics,
)


def rows(*values):
    return [RankingRow(video_id=video, score=score, clicked=clicked) for video, score, clicked in values]


def test_perfect_order_is_one_and_reverse_is_less_than_one():
    perfect = rows(("a", 0.9, 1), ("b", 0.8, 1), ("c", 0.1, 0))
    reverse = rows(("a", 0.1, 1), ("b", 0.8, 1), ("c", 0.9, 0))
    assert ndcg_at_k(perfect, 2) == pytest.approx(1.0)
    assert recall_at_k(perfect, 2) == pytest.approx(1.0)
    assert ndcg_at_k(reverse, 1) < 1.0


def test_zero_click_slate_is_excluded_and_coverage_is_reported():
    summary = summarize_ranking_metrics(
        [rows(("a", 0.5, 0)), rows(("b", 0.5, 1))], k=10
    )
    assert ndcg_at_k(rows(("a", 0.5, 0)), 1) is None
    assert summary.valid_slate_count == 1
    assert summary.total_slate_count == 2
    assert summary.coverage == pytest.approx(0.5)
    assert summary.reason == "insufficient_metric_coverage"


def test_ties_use_video_id_and_k_can_exceed_slate_length():
    tied = rows(("b", 0.5, 1), ("a", 0.5, 0))
    assert recall_at_k(tied, 1) == 0.0
    assert recall_at_k(tied, 10) == 1.0


def test_invalid_rows_and_k_fail_closed():
    with pytest.raises(ValueError, match="중복"):
        ndcg_at_k(rows(("a", 0.5, 1), ("a", 0.4, 0)), 1)
    with pytest.raises(ValueError, match="k"):
        recall_at_k(rows(("a", 0.5, 1)), 0)
