from pathlib import Path

import pytest

from autoresearch.research_harness import (
    EvaluationManifest,
    JudgeError,
    JudgePolicy,
    judge_predictions,
)


def _write_predictions(path: Path, values: str) -> None:
    path.write_text(f"row_id,prediction\n{values}", encoding="utf-8")


def _manifest() -> EvaluationManifest:
    return EvaluationManifest(labels=(("a", 1.0), ("b", 0.0)), dataset_fingerprint="f" * 64)


def test_judge_promotes_better_auc_and_preserves_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(path, "a,0.9\nb,0.1\n")

    result = judge_predictions(
        path,
        _manifest(),
        JudgePolicy(primary_metric="roc_auc", primary_min_delta=0.1),
        baseline_metrics=(("roc_auc", 0.5),),
    )

    assert result.decision == "promote"
    assert result.metrics == (("roc_auc", 1.0),)
    assert result.dataset_fingerprint == "f" * 64


def test_judge_discards_worse_or_tied_candidate(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(path, "a,0.1\nb,0.9\n")

    result = judge_predictions(
        path,
        _manifest(),
        JudgePolicy(primary_metric="roc_auc", primary_min_delta=0.0),
        baseline_metrics=(("roc_auc", 1.0),),
    )

    assert result.decision == "discard"


@pytest.mark.parametrize(
    "values,reason",
    [("a,0.9\na,0.1\n", "prediction_id_invalid"), ("a,nan\nb,0.1\n", "prediction_value_invalid")],
)
def test_judge_rejects_invalid_predictions(tmp_path: Path, values: str, reason: str) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(path, values)

    with pytest.raises(JudgeError, match=reason):
        judge_predictions(
            path,
            _manifest(),
            JudgePolicy(primary_metric="roc_auc"),
            baseline_metrics=(("roc_auc", 0.5),),
        )


def test_judge_rejects_missing_prediction_id(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(path, "a,0.9\n")

    with pytest.raises(JudgeError, match="prediction_ids_mismatch"):
        judge_predictions(
            path,
            _manifest(),
            JudgePolicy(primary_metric="roc_auc"),
            baseline_metrics=(("roc_auc", 0.5),),
        )


def test_judge_rejects_malformed_later_row(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text("row_id,prediction\na,0.9\nb,\n", encoding="utf-8")

    with pytest.raises(JudgeError, match="prediction_value_invalid"):
        judge_predictions(
            path,
            _manifest(),
            JudgePolicy(primary_metric="roc_auc"),
            baseline_metrics=(("roc_auc", 0.5),),
        )


def test_judge_rejects_duplicate_manifest_id(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_predictions(path, "a,0.9\nb,0.1\n")
    manifest = EvaluationManifest(labels=(("a", 1.0), ("a", 0.0)), dataset_fingerprint="f" * 64)

    with pytest.raises(JudgeError, match="label_id_invalid"):
        judge_predictions(
            path,
            manifest,
            JudgePolicy(primary_metric="roc_auc"),
            baseline_metrics=(("roc_auc", 0.5),),
        )
