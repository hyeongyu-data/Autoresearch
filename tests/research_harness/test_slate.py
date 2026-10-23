from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoresearch.research_harness.slate import SlateSnapshotError, build_evaluation_slate_snapshot


def _write_partition(root, day, rows):
    path = root / f"dt={day}" / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(event_id, ts, event_type, user, video, slate, rank=None, source="model"):
    return {
        "event_id": event_id,
        "event_timestamp": datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
        "event_type": event_type,
        "user_id": user,
        "video_id": video,
        "slate_id": slate,
        "rank": rank,
        "exposure_source": source,
    }


def test_builds_sealed_split_and_attributes_next_day_click(tmp_path):
    source = tmp_path / "action-log"
    _write_partition(source, "2026-10-01", [
        _row("i1", "2026-10-01T01:00:00", "impression", "u1", "v1", "s1", 1),
        _row("i2", "2026-10-01T01:01:00", "impression", "u6", "v2", "s2", 1),
    ])
    _write_partition(source, "2026-10-02", [
        _row("c1", "2026-10-02T00:10:00", "click", "u1", "v1", None),
    ])
    out = build_evaluation_slate_snapshot(
        action_log_root=source, output_root=tmp_path / "snapshots",
        output_start="2026-10-01", output_end="2026-10-01",
        slate_id_cutover_date="2026-10-01",
    )
    assert out.evaluation_id
    assert "clicked" not in pq.read_table(out.validation_slate_path).column_names
    labels = pq.read_table(out.validation_labels_path).to_pylist()
    assert {row["clicked"] for row in labels} <= {0, 1}
    assert out.manifest_path.exists()


def test_rejects_null_slate_after_cutover_and_missing_next_partition(tmp_path):
    source = tmp_path / "action-log"
    _write_partition(source, "2026-10-01", [_row("i1", "2026-10-01T01:00:00", "impression", "u1", "v1", None)])
    _write_partition(source, "2026-10-02", [_row("c1", "2026-10-02T00:10:00", "click", "u1", "v1", None)])
    with pytest.raises(SlateSnapshotError, match="null slate_id"):
        build_evaluation_slate_snapshot(action_log_root=source, output_root=tmp_path / "out", output_start="2026-10-01", output_end="2026-10-01", slate_id_cutover_date="2026-10-01")

    source2 = tmp_path / "action-log-2"
    _write_partition(source2, "2026-10-01", [_row("i1", "2026-10-01T01:00:00", "impression", "u1", "v1", "s1")])
    with pytest.raises(SlateSnapshotError, match="파티션 parquet"):
        build_evaluation_slate_snapshot(action_log_root=source2, output_root=tmp_path / "out2", output_start="2026-10-01", output_end="2026-10-01", slate_id_cutover_date="2026-10-01")
