from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path):
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
    return records


def _resolve_session_dir(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_dir():
        return path
    if path.name == "summary.json":
        return path.parent
    raise ValueError(f"Session path must be a session directory or summary.json: {path}")


def _quantile(values, q: float):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = position - lower
    return lower_value + ((upper_value - lower_value) * weight)


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _safe_rate(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return numerator / denominator


def _summarize_numeric(values, digits=4):
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": len(numeric),
        "mean": _round_or_none(sum(numeric) / len(numeric), digits),
        "min": _round_or_none(min(numeric), digits),
        "max": _round_or_none(max(numeric), digits),
        "p50": _round_or_none(_quantile(numeric, 0.50), digits),
        "p95": _round_or_none(_quantile(numeric, 0.95), digits),
    }


def _extract_count(record, explicit_key: str, list_key: str):
    if explicit_key in record and record[explicit_key] is not None:
        return int(record[explicit_key])
    values = record.get(list_key)
    if isinstance(values, list):
        return len(values)
    return 0


def build_summary(session_dir: Path):
    processed_path = session_dir / "processed_frames.jsonl"
    render_path = session_dir / "render_frames.jsonl"
    legacy_render_path = session_dir / "status_log.jsonl"

    processed_records = _load_jsonl(processed_path)
    render_records = _load_jsonl(render_path)
    if not render_records and legacy_render_path.exists():
        render_records = _load_jsonl(legacy_render_path)

    session_meta = _load_json(session_dir / "session_meta.json", {})
    runtime_config = _load_json(session_dir / "runtime_config.json", {})

    processed_frame_count = len(processed_records)
    render_frame_count = len(render_records)

    processed_invalid_count = sum(bool(record.get("invalid")) for record in processed_records)
    processed_birth_block_count = sum(
        bool(record.get("track_birth_blocked")) for record in processed_records
    )
    render_invalid_count = sum(bool(record.get("invalid")) for record in render_records)

    processed_candidate_counts = [int(record.get("candidate_count", 0)) for record in processed_records]
    processed_tracker_input_counts = [int(record.get("tracker_input_count", 0)) for record in processed_records]
    processed_confirmed_counts = [
        _extract_count(record, "confirmed_track_count", "confirmed_tracks")
        for record in processed_records
    ]
    processed_tentative_counts = [
        _extract_count(record, "tentative_track_count", "tentative_tracks")
        for record in processed_records
    ]

    render_candidate_counts = [int(record.get("candidate_count", 0)) for record in render_records]
    render_display_counts = [
        _extract_count(record, "display_track_count", "display_tracks")
        for record in render_records
    ]
    render_tentative_display_counts = [
        _extract_count(record, "tentative_display_track_count", "tentative_display_tracks")
        for record in render_records
    ]
    skipped_render_frames = [int(record.get("skipped_render_frames", 0)) for record in render_records]

    processed_multi_candidate_frames = [
        record for record in processed_records
        if int(record.get("candidate_count", 0)) >= 2
    ]
    processed_multi_confirmed_success = sum(
        1
        for record in processed_multi_candidate_frames
        if _extract_count(record, "confirmed_track_count", "confirmed_tracks") >= 2
    )

    render_multi_candidate_frames = [
        record for record in render_records
        if int(record.get("candidate_count", 0)) >= 2
    ]
    render_multi_display_success = sum(
        1
        for record in render_multi_candidate_frames
        if _extract_count(record, "display_track_count", "display_tracks") >= 2
    )

    summary = {
        "schema_version": 1,
        "summary_generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session_dir),
        "session_id": session_meta.get("session_id", session_dir.name),
        "session_meta": session_meta,
        "runtime_config_path": str(session_dir / "runtime_config.json"),
        "runtime_config": runtime_config,
        "log_files_present": {
            "processed_frames": processed_path.exists(),
            "render_frames": render_path.exists(),
            "event_log": (session_dir / "event_log.jsonl").exists(),
            "legacy_status_log": legacy_render_path.exists(),
        },
        "processed": {
            "frame_count": processed_frame_count,
            "first_frame_id": None if not processed_records else int(processed_records[0].get("frame_id", 0)),
            "last_frame_id": None if not processed_records else int(processed_records[-1].get("frame_id", 0)),
            "invalid_count": processed_invalid_count,
            "invalid_rate": _round_or_none(_safe_rate(processed_invalid_count, processed_frame_count)),
            "birth_block_count": processed_birth_block_count,
            "birth_block_rate": _round_or_none(
                _safe_rate(processed_birth_block_count, processed_frame_count)
            ),
            "capture_to_process_ms": _summarize_numeric(
                [record.get("capture_to_process_ms") for record in processed_records],
                digits=3,
            ),
            "candidate_count": _summarize_numeric(processed_candidate_counts, digits=3),
            "tracker_input_count": _summarize_numeric(processed_tracker_input_counts, digits=3),
            "confirmed_track_count": _summarize_numeric(processed_confirmed_counts, digits=3),
            "tentative_track_count": _summarize_numeric(processed_tentative_counts, digits=3),
            "multi_candidate_frame_count": len(processed_multi_candidate_frames),
            "multi_confirmed_success_rate": _round_or_none(
                _safe_rate(processed_multi_confirmed_success, len(processed_multi_candidate_frames))
            ),
        },
        "render": {
            "frame_count": render_frame_count,
            "first_frame_id": None if not render_records else int(render_records[0].get("frame_id", 0)),
            "last_frame_id": None if not render_records else int(render_records[-1].get("frame_id", 0)),
            "invalid_count": render_invalid_count,
            "invalid_rate": _round_or_none(_safe_rate(render_invalid_count, render_frame_count)),
            "capture_to_render_ms": _summarize_numeric(
                [record.get("capture_to_render_ms") for record in render_records],
                digits=3,
            ),
            "process_to_render_ms": _summarize_numeric(
                [record.get("process_to_render_ms") for record in render_records],
                digits=3,
            ),
            "display_track_count": _summarize_numeric(render_display_counts, digits=3),
            "tentative_display_track_count": _summarize_numeric(
                render_tentative_display_counts,
                digits=3,
            ),
            "candidate_count": _summarize_numeric(render_candidate_counts, digits=3),
            "skipped_render_frames": {
                "total": sum(skipped_render_frames),
                "mean_per_render": _round_or_none(
                    _safe_rate(sum(skipped_render_frames), render_frame_count),
                    digits=3,
                ),
                "max_single_render": max(skipped_render_frames) if skipped_render_frames else 0,
            },
            "multi_candidate_frame_count": len(render_multi_candidate_frames),
            "multi_display_success_rate": _round_or_none(
                _safe_rate(render_multi_display_success, len(render_multi_candidate_frames))
            ),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build summary.json from a live_motion_viewer session directory."
    )
    parser.add_argument("session", help="Path to a session directory.")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to <session>/summary.json",
    )
    args = parser.parse_args()

    session_dir = _resolve_session_dir(args.session)
    summary = build_summary(session_dir)
    output_path = Path(args.output) if args.output else session_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        from .log_html_reports import generate_reports

        generate_reports(session_dir)
    except Exception as exc:
        print(f"HTML report generation failed: {exc!r}")

    print(f"Summary written to: {output_path}")
    print(f"Processed frames: {summary['processed']['frame_count']}")
    print(f"Rendered frames: {summary['render']['frame_count']}")
    print(f"Processed invalid rate: {summary['processed']['invalid_rate']}")
    print(f"Render p95 latency: {summary['render']['capture_to_render_ms']['p95']} ms")


if __name__ == "__main__":
    main()
