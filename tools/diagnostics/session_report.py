from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from .operational_assessment import build_event_summary, build_operational_assessment


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


def _invalid_reason_counts(records):
    counts = {}
    for record in records:
        reason = str(record.get("invalid_reason") or "").strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _summarize_stage_timings(records, digits=3):
    timing_buckets = {}
    frame_count_with_timings = 0
    for record in records:
        stage_timings = record.get("stage_timings_ms")
        if not isinstance(stage_timings, dict) or not stage_timings:
            continue
        frame_count_with_timings += 1
        for stage_name, raw_value in stage_timings.items():
            if raw_value is None:
                continue
            timing_buckets.setdefault(stage_name, []).append(float(raw_value))

    timing_summary = {
        stage_name: _summarize_numeric(values, digits=digits)
        for stage_name, values in sorted(timing_buckets.items())
    }
    slowest_stage = None
    slowest_score = None
    for stage_name, stats in timing_summary.items():
        candidate_score = stats.get("p95")
        if candidate_score is None:
            candidate_score = stats.get("mean")
        if candidate_score is None:
            continue
        if slowest_score is None or candidate_score > slowest_score:
            slowest_score = candidate_score
            slowest_stage = {
                "name": stage_name,
                "p95_ms": stats.get("p95"),
                "mean_ms": stats.get("mean"),
            }

    return {
        "frame_count_with_timings": frame_count_with_timings,
        "timings": timing_summary,
        "slowest_stage": slowest_stage,
    }


def _preferred_stage_timing_summary(processed_summary, render_summary):
    def summary_score(stage_summary):
        if not isinstance(stage_summary, dict):
            return -1
        timings = stage_summary.get("timings") or {}
        return (len(timings), int(stage_summary.get("frame_count_with_timings") or 0))

    render_score = summary_score(render_summary)
    processed_score = summary_score(processed_summary)
    if render_score >= processed_score:
        selected = dict(render_summary or {})
        selected["source"] = "render"
        return selected

    selected = dict(processed_summary or {})
    selected["source"] = "processed"
    return selected


def _build_system_summary(session_dir: Path, runtime_config: dict, system_snapshot: dict):
    static_snapshot = runtime_config.get("static_snapshot") or {}
    network_config = static_snapshot.get("network") or {}
    expected_host_ip = network_config.get("host_ip")

    if not system_snapshot:
        return {
            "snapshot_present": False,
            "snapshot_path": str(session_dir / "system_snapshot.json"),
            "expected_host_ip": expected_host_ip,
            "host_ip_present": None,
            "ipv4_addresses": [],
            "power_plan_name": None,
            "power_plan_recommended": None,
            "process_priority_class": None,
            "enabled_firewall_profiles": [],
            "up_adapter_count": 0,
        }

    power = system_snapshot.get("power") or {}
    network = system_snapshot.get("network") or {}
    process = system_snapshot.get("process") or {}
    adapters = network.get("adapters") or []
    firewall_profiles = network.get("firewall_profiles") or []
    up_adapter_count = sum(
        1
        for adapter in adapters
        if str((adapter or {}).get("Status") or "").strip().lower() == "up"
    )
    enabled_firewall_profiles = [
        profile.get("Name")
        for profile in firewall_profiles
        if profile.get("Enabled") is True
    ]
    ipv4_addresses = [str(address) for address in (network.get("ipv4_addresses") or []) if address]
    host_ip_present = network.get("host_ip_present")
    if host_ip_present is None and expected_host_ip:
        host_ip_present = expected_host_ip in ipv4_addresses

    return {
        "snapshot_present": True,
        "snapshot_path": str(session_dir / "system_snapshot.json"),
        "captured_at": system_snapshot.get("captured_at"),
        "expected_host_ip": expected_host_ip,
        "host_ip_present": host_ip_present,
        "ipv4_addresses": ipv4_addresses,
        "power_plan_name": power.get("active_scheme_name"),
        "power_plan_guid": power.get("active_scheme_guid"),
        "power_plan_recommended": power.get("recommended_for_benchmarking"),
        "process_priority_class": process.get("priority_class"),
        "process_priority_class_code": process.get("priority_class_code"),
        "enabled_firewall_profiles": enabled_firewall_profiles,
        "up_adapter_count": up_adapter_count,
        "adapter_count": len(adapters),
        "numpy_version": (system_snapshot.get("python") or {}).get("numpy_version"),
        "thread_env": system_snapshot.get("env") or {},
    }


def build_summary(session_dir: Path):
    processed_path = session_dir / "processed_frames.jsonl"
    render_path = session_dir / "render_frames.jsonl"
    legacy_render_path = session_dir / "status_log.jsonl"
    event_log_path = session_dir / "event_log.jsonl"
    system_snapshot_path = session_dir / "system_snapshot.json"

    processed_records = _load_jsonl(processed_path)
    render_records = _load_jsonl(render_path)
    if not render_records and legacy_render_path.exists():
        render_records = _load_jsonl(legacy_render_path)
    events = _load_jsonl(event_log_path)

    session_meta = _load_json(session_dir / "session_meta.json", {})
    runtime_config = _load_json(session_dir / "runtime_config.json", {})
    system_snapshot = _load_json(system_snapshot_path, {})

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

    processed_stage_timings = _summarize_stage_timings(processed_records, digits=3)
    render_stage_timings = _summarize_stage_timings(render_records, digits=3)
    preferred_stage_timings = _preferred_stage_timing_summary(
        processed_stage_timings,
        render_stage_timings,
    )

    summary = {
        "schema_version": 3,
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
            "system_snapshot": system_snapshot_path.exists(),
        },
        "processed": {
            "frame_count": processed_frame_count,
            "first_frame_id": None if not processed_records else int(processed_records[0].get("frame_id", 0)),
            "last_frame_id": None if not processed_records else int(processed_records[-1].get("frame_id", 0)),
            "invalid_count": processed_invalid_count,
            "invalid_rate": _round_or_none(_safe_rate(processed_invalid_count, processed_frame_count)),
            "invalid_reason_counts": _invalid_reason_counts(processed_records),
            "birth_block_count": processed_birth_block_count,
            "birth_block_rate": _round_or_none(
                _safe_rate(processed_birth_block_count, processed_frame_count)
            ),
            "max_udp_gap_count": max((int(record.get("udp_gap_count", 0)) for record in processed_records), default=0),
            "max_out_of_sequence_count": max(
                (int(record.get("out_of_sequence_count", 0)) for record in processed_records),
                default=0,
            ),
            "max_byte_mismatch_count": max(
                (int(record.get("byte_mismatch_count", 0)) for record in processed_records),
                default=0,
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
            "stage_timings_ms": processed_stage_timings,
        },
        "render": {
            "frame_count": render_frame_count,
            "first_frame_id": None if not render_records else int(render_records[0].get("frame_id", 0)),
            "last_frame_id": None if not render_records else int(render_records[-1].get("frame_id", 0)),
            "invalid_count": render_invalid_count,
            "invalid_rate": _round_or_none(_safe_rate(render_invalid_count, render_frame_count)),
            "invalid_reason_counts": _invalid_reason_counts(render_records),
            "max_udp_gap_count": max((int(record.get("udp_gap_count", 0)) for record in render_records), default=0),
            "max_out_of_sequence_count": max(
                (int(record.get("out_of_sequence_count", 0)) for record in render_records),
                default=0,
            ),
            "max_byte_mismatch_count": max(
                (int(record.get("byte_mismatch_count", 0)) for record in render_records),
                default=0,
            ),
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
            "stage_timings_ms": render_stage_timings,
        },
        "system": _build_system_summary(session_dir, runtime_config, system_snapshot),
        "diagnostics": {
            "preferred_stage_timings_ms": preferred_stage_timings,
        },
    }
    summary["event"] = build_event_summary(events)
    summary["assessment"] = build_operational_assessment(summary, summary["event"])
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
