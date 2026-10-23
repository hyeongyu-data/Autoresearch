"""action log에서 평가 slate와 봉인 labels를 만드는 스냅샷 빌더.

[파이프라인] 원천 action log의 파티션을 시간 경계로 읽어 validation/final holdout
입력과 정답을 분리하고 content-addressed 경로에 한 번 게시한다.

[기능] ``slate_id`` cutover 검증, 30분 click 귀속, 사용자 단위 분할, manifest 생성,
로컬 parquet 읽기와 write-once 게시를 제공한다.

[비책임] candidate 코드 실행, 지표 계산, Judge/Controller, 원격 자격 증명 관리는
담당하지 않는다. labels는 이 모듈의 출력이지만 candidate workspace에 전달하는 일은
호출자가 담당한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


class SlateSnapshotError(ValueError):
    """평가 snapshot 계약을 만족하지 못한 경우의 fail-closed 오류."""


@dataclass(frozen=True, slots=True)
class EvaluationSlateSnapshot:
    """게시된 validation/final 산출물과 manifest 위치."""

    evaluation_id: str
    root: Path
    manifest_path: Path
    validation_slate_path: Path
    validation_labels_path: Path
    final_slate_path: Path
    final_labels_path: Path


_REQUIRED = {"event_timestamp", "event_type", "user_id", "video_id", "slate_id"}
_SALT = "research-harness-slate-v1:"
_KST = timezone(timedelta(hours=9))


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise SlateSnapshotError(f"날짜 형식이 잘못되었습니다: {value}") from error


def _partition_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise SlateSnapshotError("평가 종료일은 시작일보다 빠를 수 없습니다")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _local_partition_files(root: Path, partition: date) -> list[Path]:
    directory = root / f"dt={partition:%Y-%m-%d}"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.parquet") if path.is_file())


def _read_rows(root: str | Path, partitions: Iterable[date]) -> list[dict[str, object]]:
    """파티션 parquet를 읽는다. GCS는 pyarrow의 URI reader를 그대로 사용한다."""
    rows: list[dict[str, object]] = []
    root_text = str(root)
    if root_text.startswith("gs://"):
        from pyarrow.fs import FileSelector, GcsFileSystem

        filesystem = GcsFileSystem()
        base = root_text[5:].rstrip("/")
        for partition in partitions:
            prefix = f"{base}/dt={partition:%Y-%m-%d}"
            infos = filesystem.get_file_info(FileSelector(prefix, recursive=True))
            files = [info.path for info in infos if info.is_file and info.path.endswith(".parquet")]
            if not files:
                raise SlateSnapshotError(f"파티션 parquet가 없습니다: dt={partition:%Y-%m-%d}")
            for path in sorted(files):
                rows.extend(pq.read_table(path, filesystem=filesystem).to_pylist())
        return rows

    base = Path(root)
    for partition in partitions:
        files = _local_partition_files(base, partition)
        if not files:
            raise SlateSnapshotError(f"파티션 parquet가 없습니다: dt={partition:%Y-%m-%d}")
        for path in files:
            rows.extend(pq.read_table(path).to_pylist())
    return rows


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise SlateSnapshotError("event_timestamp가 datetime 또는 ISO 문자열이 아닙니다")


def _event_dt(value: object) -> date:
    return _timestamp(value).astimezone(_KST).date()


def _user_bucket(user_id: str) -> int:
    digest = hashlib.sha256((_SALT + user_id).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % 10


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SlateSnapshotError(f"write-once 대상이 이미 다른 내용으로 존재합니다: {path}")
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise SlateSnapshotError(f"동시 게시 내용이 다릅니다: {path}")
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _parquet_bytes(rows: list[dict[str, object]], schema: pa.Schema) -> bytes:
    table = pa.Table.from_pylist(rows, schema=schema)
    fd, name = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, name, compression="snappy")
        return Path(name).read_bytes()
    finally:
        Path(name).unlink(missing_ok=True)


_SLATE_SCHEMA = pa.schema([
    ("evaluation_id", pa.string()), ("slate_id", pa.string()), ("user_id", pa.string()),
    ("video_id", pa.string()), ("event_timestamp", pa.timestamp("us", tz="UTC")),
    ("original_rank", pa.int64()), ("candidate_source", pa.string()),
])
_LABEL_SCHEMA = pa.schema([
    ("evaluation_id", pa.string()), ("slate_id", pa.string()), ("video_id", pa.string()),
    ("clicked", pa.int8()),
])


def build_evaluation_slate_snapshot(
    *,
    action_log_root: str | Path,
    output_root: Path,
    output_start: date | str,
    output_end: date | str,
    slate_id_cutover_date: date | str,
    history_start: date | str | None = None,
) -> EvaluationSlateSnapshot:
    """평가 기간의 slate/labels를 분리해 결정적으로 게시한다."""
    start = _as_date(output_start)
    end = _as_date(output_end)
    cutover = _as_date(slate_id_cutover_date)
    if cutover > start:
        raise SlateSnapshotError("slate_id_cutover_date는 평가 시작일보다 늦을 수 없습니다")
    output_dates = _partition_dates(start, end)
    scan_dates = _partition_dates(start, end + timedelta(days=1))
    history_begin = _as_date(history_start) if history_start is not None else None
    if history_begin is not None and history_begin >= start:
        raise SlateSnapshotError("history 시작일은 평가 시작일보다 빨라야 합니다")

    rows = _read_rows(action_log_root, scan_dates)
    if not rows:
        raise SlateSnapshotError("action log가 비어 있습니다")
    if any(not _REQUIRED.issubset(row) for row in rows):
        raise SlateSnapshotError("action log에 필수 컬럼이 없습니다")
    impressions: list[dict[str, object]] = []
    clicks: list[dict[str, object]] = []
    for row in rows:
        dt = _event_dt(row["event_timestamp"])
        if row["event_type"] == "impression" and dt >= cutover and row["slate_id"] is None:
            raise SlateSnapshotError(f"cutover 이후 null slate_id가 있습니다: dt={dt}")
        if row["event_type"] == "impression" and dt in output_dates:
            impressions.append(row)
        if row["event_type"] == "click" and dt in scan_dates:
            clicks.append(row)
    if not impressions:
        raise SlateSnapshotError("평가 출력 구간에 impression이 없습니다")

    latest: dict[tuple[str, str], list[tuple[datetime, dict[str, object]]]] = {}
    for row in impressions + [r for r in rows if r.get("event_type") == "impression" and _event_dt(r["event_timestamp"]) == end + timedelta(days=1)]:
        key = (str(row["user_id"]), str(row["video_id"]))
        latest.setdefault(key, []).append((_timestamp(row["event_timestamp"]), row))
    for values in latest.values():
        values.sort(key=lambda item: item[0])
    clicked_keys: set[tuple[str, str]] = set()
    for click in clicks:
        click_time = _timestamp(click["event_timestamp"])
        candidates = latest.get((str(click["user_id"]), str(click["video_id"])), [])
        prior = [item for item in candidates if item[0] < click_time and click_time - item[0] <= timedelta(minutes=30)]
        if prior:
            source = prior[-1][1]
            if _event_dt(source["event_timestamp"]) in output_dates:
                clicked_keys.add((str(source["slate_id"]), str(source["video_id"])))

    split_rows: dict[str, list[dict[str, object]]] = {"validation": [], "final_holdout": []}
    split_labels: dict[str, list[dict[str, object]]] = {"validation": [], "final_holdout": []}
    for row in sorted(impressions, key=lambda item: (str(item["slate_id"]), str(item["user_id"]), str(item["video_id"]))):
        user_id = str(row["user_id"])
        split = "validation" if _user_bucket(user_id) < 8 else "final_holdout"
        evaluation_id = "pending"
        slate_row = {
            "evaluation_id": evaluation_id, "slate_id": str(row["slate_id"]), "user_id": user_id,
            "video_id": str(row["video_id"]), "event_timestamp": _timestamp(row["event_timestamp"]).astimezone(timezone.utc),
            "original_rank": row.get("rank"), "candidate_source": row.get("exposure_source"),
        }
        split_rows[split].append(slate_row)
        split_labels[split].append({"evaluation_id": evaluation_id, "slate_id": str(row["slate_id"]), "video_id": str(row["video_id"]), "clicked": int((str(row["slate_id"]), str(row["video_id"])) in clicked_keys)})
    if not split_rows["validation"] or not split_rows["final_holdout"]:
        raise SlateSnapshotError("validation/final holdout 중 비어 있는 split이 있습니다")

    content = _parquet_bytes(split_rows["validation"], _SLATE_SCHEMA) + _parquet_bytes(split_labels["validation"], _LABEL_SCHEMA) + _parquet_bytes(split_rows["final_holdout"], _SLATE_SCHEMA) + _parquet_bytes(split_labels["final_holdout"], _LABEL_SCHEMA)
    evaluation_id = hashlib.sha256(content).hexdigest()
    for split in split_rows:
        for row in split_rows[split]:
            row["evaluation_id"] = evaluation_id
        for row in split_labels[split]:
            row["evaluation_id"] = evaluation_id
    root = output_root / "by-hash" / evaluation_id
    files = {
        "validation/slate.parquet": _parquet_bytes(split_rows["validation"], _SLATE_SCHEMA),
        "validation/labels.parquet": _parquet_bytes(split_labels["validation"], _LABEL_SCHEMA),
        "final_holdout/slate.parquet": _parquet_bytes(split_rows["final_holdout"], _SLATE_SCHEMA),
        "final_holdout/labels.parquet": _parquet_bytes(split_labels["final_holdout"], _LABEL_SCHEMA),
    }
    for relative, payload in files.items():
        _write_once(root / relative, payload)
    manifest = {
        "manifest_version": "evaluation_slate_snapshot_v1", "evaluation_id": evaluation_id,
        "output_start": start.isoformat(), "output_end": end.isoformat(),
        "slate_id_cutover_date": cutover.isoformat(), "history_start": history_begin.isoformat() if history_begin else None,
        "history_partition_upper_bound": (start - timedelta(days=1)).isoformat(),
        "complete_label_upper_bound": (start - timedelta(days=2)).isoformat(),
        "scan_partitions": [day.isoformat() for day in scan_dates],
        "split_rule": "sha256(research-harness-slate-v1: + user_id) % 10; 0..7 validation, 8..9 final_holdout",
        "row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "optional_non_null_ratio": {"original_rank": sum(row["original_rank"] is not None for row in split_rows["validation"]) / len(split_rows["validation"]), "candidate_source": sum(row["candidate_source"] is not None for row in split_rows["validation"]) / len(split_rows["validation"])},
    }
    manifest_path = root / "manifest.json"
    _write_once(manifest_path, _canonical_json(manifest))
    return EvaluationSlateSnapshot(evaluation_id, root, manifest_path, root / "validation/slate.parquet", root / "validation/labels.parquet", root / "final_holdout/slate.parquet", root / "final_holdout/labels.parquet")
