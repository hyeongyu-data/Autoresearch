"""평가 slate 내부의 리랭킹 지표를 계산하는 순수 함수.

[파이프라인] 봉인된 slate·labels와 candidate score 사이에서 결정적 순위를 만들고
slate 단위 NDCG/Recall과 유효 slate coverage를 제공한다.

[기능] 이진 click relevance, score 내림차순·video_id 오름차순 tie-break, 0-click
slate 제외, coverage 부족 상태를 계산한다.

[비책임] 모델 학습·확률 보정·Judge/Controller 배선·파일과 네트워크 I/O는 담당하지
않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


MIN_VALID_SLATES = 30
MIN_COVERAGE = 0.20


@dataclass(frozen=True, slots=True)
class RankingRow:
    """한 slate 안의 영상 점수와 봉인된 click relevance."""

    video_id: str
    score: float
    clicked: int


@dataclass(frozen=True, slots=True)
class RankingMetricSummary:
    """slate 평균 지표와 데이터 충분성 근거."""

    ndcg: float | None
    recall: float | None
    valid_slate_count: int
    total_slate_count: int
    coverage: float
    reason: str | None


def _ordered(rows: Sequence[RankingRow]) -> list[RankingRow]:
    if not rows:
        return []
    if len({row.video_id for row in rows}) != len(rows):
        raise ValueError("한 slate 안의 video_id는 중복될 수 없습니다")
    for row in rows:
        if row.clicked not in (0, 1):
            raise ValueError("clicked는 0 또는 1이어야 합니다")
        if not math.isfinite(row.score):
            raise ValueError("score는 유한한 수여야 합니다")
    return sorted(rows, key=lambda row: (-row.score, row.video_id))


def _require_k(k: int) -> None:
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다")


def ndcg_at_k(rows: Sequence[RankingRow], k: int) -> float | None:
    """클릭 이진 relevance의 slate 단위 NDCG@k를 계산한다.

    클릭이 한 건도 없는 slate는 지표가 정의되지 않으므로 ``None``을 반환한다.
    """
    _require_k(k)
    ordered = _ordered(rows)
    positives = sum(row.clicked for row in ordered)
    if positives == 0:
        return None

    def dcg(relevances: Sequence[int]) -> float:
        return sum(relevance / math.log2(index + 2) for index, relevance in enumerate(relevances))

    observed = dcg([row.clicked for row in ordered[:k]])
    ideal = dcg(sorted((row.clicked for row in ordered), reverse=True)[:k])
    return observed / ideal if ideal else None


def recall_at_k(rows: Sequence[RankingRow], k: int) -> float | None:
    """클릭 이진 relevance의 slate 단위 Recall@k를 계산한다."""
    _require_k(k)
    ordered = _ordered(rows)
    positives = sum(row.clicked for row in ordered)
    if positives == 0:
        return None
    return sum(row.clicked for row in ordered[:k]) / positives


def summarize_ranking_metrics(
    slates: Iterable[Sequence[RankingRow]], *, k: int
) -> RankingMetricSummary:
    """여러 slate의 macro 평균과 coverage 게이트를 계산한다."""
    _require_k(k)
    total = 0
    valid = 0
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    for rows in slates:
        total += 1
        ndcg = ndcg_at_k(rows, k)
        recall = recall_at_k(rows, k)
        if ndcg is None or recall is None:
            continue
        valid += 1
        ndcg_values.append(ndcg)
        recall_values.append(recall)

    coverage = valid / total if total else 0.0
    reason = None
    if valid < MIN_VALID_SLATES or coverage < MIN_COVERAGE:
        reason = "insufficient_metric_coverage"
    return RankingMetricSummary(
        ndcg=sum(ndcg_values) / valid if valid else None,
        recall=sum(recall_values) / valid if valid else None,
        valid_slate_count=valid,
        total_slate_count=total,
        coverage=coverage,
        reason=reason,
    )
