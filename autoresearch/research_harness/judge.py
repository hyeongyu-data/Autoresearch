"""후보 예측을 후보 코드와 독립적으로 채점하는 순수 Judge.

[파이프라인] candidate가 학습·예측을 끝내고 `predictions.csv`를 제출한 다음부터
결과 report에 판정값을 넘기기 전까지를 담당한다. 정답과 정책을 읽어 입력 계약을
검증하고 primary metric과 guardrail을 계산한다.

[기능] row ID의 일치·중복·누락, 유한한 예측값, 지원 지표를 검증한다. 현재는
ROC-AUC와 RMSE를 표준 라이브러리만으로 계산하고 primary 방향과 guardrail 하한을
적용한다. 후보 모듈·학습 모델·평가 구현을 import하거나 실행하지 않는다.

[비책임] 후보 workspace 실행·컨테이너 격리·데이터 생성·trial ledger·Kubernetes
Job·운영 승격은 담당하지 않는다. 파일 시스템 밖의 완전한 보안 경계는 호출자가
제공해야 한다.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MetricName = Literal["roc_auc", "rmse"]


class JudgeError(ValueError):
    """예측 또는 평가 계약이 잘못된 경우의 fail-closed 오류."""


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """Judge가 소유하는 정답과 평가 대상의 불변 표현."""

    labels: tuple[tuple[str, float], ...]
    dataset_fingerprint: str


@dataclass(frozen=True, slots=True)
class JudgePolicy:
    """primary metric과 guardrail의 판정 정책."""

    primary_metric: MetricName
    primary_min_delta: float = 0.0
    guardrails: tuple[tuple[MetricName, float], ...] = ()


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """독립 Judge가 반환하는 재현 가능한 판정 결과."""

    decision: Literal["promote", "discard"]
    metrics: tuple[tuple[str, float], ...]
    dataset_fingerprint: str
    prediction_count: int


def _read_predictions(path: Path) -> dict[str, float]:
    """`row_id,prediction` CSV를 읽어 중복·누락 검사의 입력으로 만든다."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError) as error:
        raise JudgeError("predictions_unreadable") from error
    if not rows or any(set(row) != {"row_id", "prediction"} for row in rows):
        raise JudgeError("prediction_schema_invalid")
    predictions: dict[str, float] = {}
    for row in rows:
        row_id = (row.get("row_id") or "").strip()
        if not row_id or row_id in predictions:
            raise JudgeError("prediction_id_invalid")
        try:
            value = float(row["prediction"])
        except (TypeError, ValueError) as error:
            raise JudgeError("prediction_value_invalid") from error
        if not math.isfinite(value):
            raise JudgeError("prediction_value_invalid")
        predictions[row_id] = value
    return predictions


def _roc_auc(labels: list[float], scores: list[float]) -> float:
    positives = sum(label == 1.0 for label in labels)
    negatives = sum(label == 0.0 for label in labels)
    if positives == 0 or negatives == 0:
        raise JudgeError("roc_auc_class_missing")
    concordant = 0.0
    for index, label in enumerate(labels):
        if label != 1.0:
            continue
        for other_index, other in enumerate(labels):
            if other != 0.0:
                continue
            if scores[index] > scores[other_index]:
                concordant += 1.0
            elif scores[index] == scores[other_index]:
                concordant += 0.5
    return concordant / (positives * negatives)


def _rmse(labels: list[float], scores: list[float]) -> float:
    return math.sqrt(sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels))


def _metric(metric: MetricName, labels: list[float], scores: list[float]) -> float:
    if metric == "roc_auc":
        return _roc_auc(labels, scores)
    if metric == "rmse":
        return _rmse(labels, scores)
    raise JudgeError("metric_unsupported")


def judge_predictions(
    predictions_path: Path,
    manifest: EvaluationManifest,
    policy: JudgePolicy,
    *,
    baseline_metrics: tuple[tuple[str, float], ...] = (),
) -> JudgeResult:
    """후보 예측을 검증하고 baseline 대비 독립 판정을 반환한다.

    `baseline_metrics`는 후보가 계산한 값이 아니라 Judge가 같은 정책으로 보관한
    값이어야 한다. 이 함수는 그 출처를 확인할 수 없으므로 호출 경계에서 Judge
    소유 산출물만 전달해야 한다.
    """
    predictions = _read_predictions(predictions_path)
    expected: dict[str, float] = {}
    for row_id, label in manifest.labels:
        if not row_id or row_id in expected:
            raise JudgeError("label_id_invalid")
        expected[row_id] = label
    if set(predictions) != set(expected):
        raise JudgeError("prediction_ids_mismatch")
    labels = [expected[row_id] for row_id in expected]
    scores = [predictions[row_id] for row_id in expected]
    if any(label not in (0.0, 1.0) for label in labels):
        raise JudgeError("label_invalid")

    requested = (policy.primary_metric, *(metric for metric, _ in policy.guardrails))
    metrics = tuple((metric, _metric(metric, labels, scores)) for metric in dict.fromkeys(requested))
    baseline = dict(baseline_metrics)
    if policy.primary_metric not in baseline:
        raise JudgeError("baseline_metric_missing")

    current = dict(metrics)
    primary_delta = current[policy.primary_metric] - baseline[policy.primary_metric]
    if policy.primary_metric == "rmse":
        primary_delta = -primary_delta
    decision = "promote" if primary_delta >= policy.primary_min_delta else "discard"
    for metric, minimum_delta in policy.guardrails:
        if metric not in baseline:
            raise JudgeError("baseline_metric_missing")
        delta = current[metric] - baseline[metric]
        if metric == "rmse":
            delta = -delta
        if delta < minimum_delta:
            decision = "discard"
    return JudgeResult(
        decision=decision,
        metrics=metrics,
        dataset_fingerprint=manifest.dataset_fingerprint,
        prediction_count=len(predictions),
    )
