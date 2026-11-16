"""candidate 예측 파일을 봉인된 경계에서 검증·채점하는 Sealed Judge.

[파이프라인] candidate가 예측 파일을 제출한 뒤, Judge가 원본 labels와 안전한 사본만
읽어 schema·지표·승격 판정을 반환하는 구간을 담당한다.

[기능] 단일 FD 복사, CSV 계약 검증, slate ranking 및 확률 지표, σ 기반 결정을 제공한다.
[비책임] candidate 코드 실행, 학습, sandbox/컨테이너 격리, Controller·클라우드 배선은
담당하지 않는다.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from autoresearch.research_harness.ranking_metrics import RankingRow, ndcg_at_k, recall_at_k


MetricName = str
Decision = Literal["promote", "revise", "discard"]
MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 300_000
MAX_ID_BYTES = 64
MAX_SCORE_BYTES = 24


class JudgeError(ValueError):
    """예측 또는 평가 계약이 잘못된 경우의 fail-closed 오류."""


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """Judge가 소유하는 labels의 불변 표현.

    새 계약은 ``evaluation_id, slate_id, video_id, clicked`` 4-tuple이며, 이전
    ``row_id, label`` 2-tuple도 호환을 위해 읽을 수 있다.
    """

    labels: tuple[tuple[object, ...], ...]
    dataset_fingerprint: str


@dataclass(frozen=True, slots=True)
class JudgePolicy:
    """primary·guardrail 지표와 σ 기반 판정 정책."""

    primary_metric: MetricName
    primary_min_delta: float = 0.0
    guardrails: tuple[tuple[MetricName, float], ...] = ()
    sigma_by_metric: tuple[tuple[MetricName, float], ...] = ()


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """독립 Judge가 반환하는 재현 가능한 판정 결과."""

    decision: Decision
    metrics: tuple[tuple[str, float], ...]
    dataset_fingerprint: str
    prediction_count: int
    reason_code: str | None = None


def _copy_once(source: Path) -> Path:
    """원본을 O_NOFOLLOW FD에서 한 번 읽어 제한된 임시 사본으로 복사한다."""
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(source, flags)
    except OSError as error:
        raise JudgeError("invalid_predictions") from error
    try:
        before = os.fstat(fd)
        if before.st_size > MAX_BYTES:
            raise JudgeError("invalid_predictions")
        with tempfile.NamedTemporaryFile(prefix="judge-predictions-", delete=False) as stream:
            destination = Path(stream.name)
            remaining = MAX_BYTES + 1
            while remaining:
                chunk = os.read(fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                stream.write(chunk)
                remaining -= len(chunk)
                if remaining == 0:
                    raise JudgeError("invalid_predictions")
            stream.flush()
            os.fsync(stream.fileno())
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns
        ):
            destination.unlink(missing_ok=True)
            raise JudgeError("invalid_predictions")
        return destination
    except Exception:
        if "destination" in locals():
            destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)


def _read_predictions(path: Path) -> tuple[dict[tuple[str, ...], float], bool]:
    copied = _copy_once(path)
    try:
        with copied.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            sealed = fields == ("evaluation_id", "slate_id", "video_id", "score")
            legacy = fields == ("row_id", "prediction")
            if not sealed and not legacy:
                raise JudgeError("invalid_predictions")
            values: dict[tuple[str, ...], float] = {}
            count = 0
            for row in reader:
                count += 1
                if count > MAX_ROWS:
                    raise JudgeError("invalid_predictions")
                if sealed:
                    key = tuple((row.get(name) or "") for name in ("evaluation_id", "slate_id", "video_id"))
                    tokens = key
                else:
                    key = ((row.get("row_id") or ""),)
                    tokens = key
                if any(not token or any(ord(char) > 127 or char in ',"\r\n' for char in token) or len(token.encode()) > MAX_ID_BYTES for token in tokens):
                    raise JudgeError("prediction_id_invalid" if not sealed else "invalid_predictions")
                token = row.get("score" if sealed else "prediction") or ""
                if len(token.encode()) > MAX_SCORE_BYTES:
                    raise JudgeError("prediction_value_invalid" if not sealed else "invalid_predictions")
                try:
                    score = float(token)
                except ValueError as error:
                    raise JudgeError("prediction_value_invalid" if not sealed else "invalid_predictions") from error
                if not math.isfinite(score) or (sealed and not 0 <= score <= 1):
                    raise JudgeError("prediction_value_invalid" if not sealed else "invalid_predictions")
                if key in values:
                    raise JudgeError("prediction_id_invalid" if not sealed else "invalid_predictions")
                values[key] = score
            if not values:
                raise JudgeError("invalid_predictions")
            return values, sealed
    except (OSError, UnicodeError) as error:
        raise JudgeError("invalid_predictions") from error
    finally:
        copied.unlink(missing_ok=True)


def _legacy_metrics(labels: list[float], scores: list[float], metric: str) -> float:
    if metric == "roc_auc":
        positives = sum(label == 1.0 for label in labels)
        negatives = sum(label == 0.0 for label in labels)
        if positives == 0 or negatives == 0:
            raise JudgeError("metric_unavailable")
        concordant = 0.0
        for index, label in enumerate(labels):
            if label != 1.0:
                continue
            for other_index, other in enumerate(labels):
                if other != 0.0:
                    continue
                concordant += scores[index] > scores[other_index] or 0.5 * (scores[index] == scores[other_index])
        return concordant / (positives * negatives)
    if metric == "rmse":
        return math.sqrt(sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels))
    raise JudgeError("metric_unsupported")


def _probability_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if not labels or len(set(labels)) < 2:
        raise JudgeError("metric_unavailable")
    epsilon = 1e-15
    logloss = -sum(label * math.log(max(epsilon, min(1 - epsilon, score))) + (1 - label) * math.log(max(epsilon, min(1 - epsilon, 1 - score))) for label, score in zip(labels, scores)) / len(labels)
    brier = sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)
    return {"log_loss": logloss, "brier": brier}


def _sealed_metrics(manifest: EvaluationManifest, predictions: dict[tuple[str, ...], float]) -> dict[str, float]:
    rows = [(str(e), str(s), str(v), int(c)) for e, s, v, c in manifest.labels]
    expected = {(e, s, v): c for e, s, v, c in rows}
    if len(expected) != len(rows):
        raise JudgeError("label_id_invalid")
    if set(predictions) != set(expected):
        raise JudgeError("invalid_predictions")
    slates: dict[str, list[RankingRow]] = {}
    labels: list[int] = []
    scores: list[float] = []
    for key, clicked in expected.items():
        row = RankingRow(video_id=key[2], score=predictions[key], clicked=clicked)
        slates.setdefault(key[1], []).append(row)
        labels.append(clicked)
        scores.append(predictions[key])
    valid = [slate for slate in slates.values() if any(row.clicked for row in slate)]
    if not valid or len(valid) / len(slates) < 0.20 or len(valid) < 30:
        raise JudgeError("insufficient_metric_coverage")
    metrics: dict[str, float] = {}
    for name, k, fn in (("ndcg_at_10", 10, ndcg_at_k), ("recall_at_10", 10, recall_at_k), ("ndcg_at_24", 24, ndcg_at_k)):
        values = [fn(slate, k) for slate in valid]
        metrics[name] = sum(value for value in values if value is not None) / len(values)
    metrics.update(_probability_metrics(labels, scores))
    return metrics


def judge_predictions(predictions_path: Path, manifest: EvaluationManifest, policy: JudgePolicy, *, baseline_metrics: tuple[tuple[str, float], ...] = ()) -> JudgeResult:
    """예측을 검증하고 legacy 또는 sealed 계약으로 판정한다."""
    predictions, sealed = _read_predictions(predictions_path)
    if sealed:
        metrics = _sealed_metrics(manifest, predictions)
        baseline = dict(baseline_metrics)
        sigma = dict(policy.sigma_by_metric)
        requested = {policy.primary_metric, *(metric for metric, _ in policy.guardrails), "recall_at_10", "ndcg_at_24"}
        if any(metric not in metrics for metric in requested):
            return JudgeResult("revise", tuple(metrics.items()), manifest.dataset_fingerprint, len(predictions), "metric_unavailable")
        if any(metric not in baseline or metric not in sigma or sigma[metric] <= 1e-6 for metric in requested):
            raise JudgeError("metric_sigma_invalid")
        primary_delta = (metrics[policy.primary_metric] - baseline[policy.primary_metric]) / sigma[policy.primary_metric]
        if primary_delta < 2:
            return JudgeResult("discard", tuple(metrics.items()), manifest.dataset_fingerprint, len(predictions), "primary_below_2sigma")
        for metric in requested - {policy.primary_metric}:
            if (metrics[metric] - baseline[metric]) / sigma[metric] < -1:
                return JudgeResult("discard", tuple(metrics.items()), manifest.dataset_fingerprint, len(predictions), "guardrail_below_minus_1sigma")
        return JudgeResult("promote", tuple(metrics.items()), manifest.dataset_fingerprint, len(predictions))

    if len({str(row[0]) for row in manifest.labels}) != len(manifest.labels):
        raise JudgeError("label_id_invalid")
    expected = {tuple((str(row[0]),)): float(row[1]) for row in manifest.labels}
    if set(predictions) != set(expected):
        raise JudgeError("prediction_ids_mismatch")
    labels = list(expected.values())
    scores = [predictions[key] for key in expected]
    metrics = {metric: _legacy_metrics(labels, scores, metric) for metric in dict.fromkeys((policy.primary_metric, *(metric for metric, _ in policy.guardrails)))}
    baseline = dict(baseline_metrics)
    if policy.primary_metric not in baseline:
        raise JudgeError("baseline_metric_missing")
    def delta(metric: str) -> float:
        value = metrics[metric] - baseline[metric]
        return -value if metric in {"rmse", "log_loss", "brier"} else value
    decision: Decision = "promote" if delta(policy.primary_metric) >= policy.primary_min_delta else "discard"
    for metric, minimum in policy.guardrails:
        if metric not in baseline or delta(metric) < minimum:
            decision = "discard"
    return JudgeResult(decision, tuple(metrics.items()), manifest.dataset_fingerprint, len(predictions))
