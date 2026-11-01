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


def _sealed_manifest() -> EvaluationManifest:
    labels = []
    for index in range(30):
        labels.extend((("eval", f"s{index}", f"a{index}", 1.0), ("eval", f"s{index}", f"b{index}", 0.0)))
    return EvaluationManifest(labels=tuple(labels), dataset_fingerprint="e" * 64)


def test_sealed_judge_ranking_metrics_and_sigma_decision(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    lines = ["evaluation_id,slate_id,video_id,score"]
    for index in range(30):
        lines.extend((f"eval,s{index},a{index},0.9", f"eval,s{index},b{index},0.1"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = judge_predictions(
        path,
        _sealed_manifest(),
        JudgePolicy(
            primary_metric="ndcg_at_10",
            sigma_by_metric=(
                ("ndcg_at_10", 0.1),
                ("recall_at_10", 0.1),
                ("ndcg_at_24", 0.1),
            ),
        ),
        baseline_metrics=(("ndcg_at_10", 0.0), ("recall_at_10", 0.0), ("ndcg_at_24", 0.0)),
    )
    assert result.decision == "promote"
    assert dict(result.metrics)["ndcg_at_10"] == 1.0


def test_sealed_judge_rejects_wrong_schema_and_out_of_range_score(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text("evaluation_id,slate_id,video_id,score\neval,s1,a,2\n", encoding="utf-8")
    with pytest.raises(JudgeError, match="invalid_predictions"):
        judge_predictions(path, _sealed_manifest(), JudgePolicy(primary_metric="ndcg_at_10"))


def test_judge_rejects_prediction_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.csv"
    _write_predictions(target, "a,0.9\nb,0.1\n")
    link = tmp_path / "predictions.csv"
    link.symlink_to(target)
    with pytest.raises(JudgeError, match="invalid_predictions"):
        judge_predictions(link, _manifest(), JudgePolicy(primary_metric="roc_auc"), baseline_metrics=(("roc_auc", 0.5),))
