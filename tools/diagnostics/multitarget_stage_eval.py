"""Summarize multi-target stage-cache behavior for a fixed replay/live session.

This tool is intentionally read-only for raw/session logs. It consumes
``lab_data/stage_cache/<session-or-mode-key>/frame_trace.jsonl`` and ``frames.jsonl`` and
writes a compact report that shows where likely multi-person evidence collapses
from detection candidates to DBSCAN, tracker, or display output.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stage_cache_key(session_id: str, mode: str | None) -> str:
    normalized = str(mode or "baseline").strip().lower()
    if normalized in {"", "baseline"}:
        return str(session_id)
    return f"{session_id}__{normalized}"


def _get(payload: dict, *keys, default=None):
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _candidate_strength(candidate: dict) -> float:
    score = _as_float(candidate.get("score"), 0.0)
    if score > 0:
        return score
    rdi_peak = _as_float(candidate.get("rdi_peak"), 0.0)
    if rdi_peak > 0:
        return rdi_peak
    return _as_float(candidate.get("rai_peak"), 0.0)


def _candidate_distance(a: dict, b: dict) -> float:
    ax = _as_float(a.get("x_m"), math.nan)
    ay = _as_float(a.get("y_m"), math.nan)
    bx = _as_float(b.get("x_m"), math.nan)
    by = _as_float(b.get("y_m"), math.nan)
    if any(math.isnan(v) for v in [ax, ay, bx, by]):
        return 0.0
    return math.hypot(ax - bx, ay - by)


def _doppler_distance(a: dict, b: dict) -> int:
    return abs(_as_int(a.get("doppler_bin"), 0) - _as_int(b.get("doppler_bin"), 0))


def _independent_candidate_count(
    candidates: list[dict],
    *,
    min_separation_m: float,
    min_doppler_bins: int,
    min_score_ratio: float,
) -> tuple[int, list[dict]]:
    if not candidates:
        return 0, []
    sorted_candidates = sorted(candidates, key=_candidate_strength, reverse=True)
    best_strength = max(_candidate_strength(sorted_candidates[0]), 1e-9)
    selected: list[dict] = []
    for candidate in sorted_candidates:
        if _candidate_strength(candidate) < best_strength * float(min_score_ratio):
            continue
        if not selected:
            selected.append(candidate)
            continue
        independent = True
        for kept in selected:
            cart_distance = _candidate_distance(candidate, kept)
            doppler_distance = _doppler_distance(candidate, kept)
            if cart_distance < min_separation_m and doppler_distance < min_doppler_bins:
                independent = False
                break
        if independent:
            selected.append(candidate)
    return len(selected), selected


def _count_distribution(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(_as_int(row.get(key), 0))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _frame_counts(trace: dict, frame: dict) -> dict:
    detection = trace.get("detection") or {}
    tracker = trace.get("tracker") or {}
    lifecycle = _get(tracker, "track_lifecycle", default={}) or {}
    display = trace.get("display_output") or {}
    return {
        "ordinal": _as_int(trace.get("ordinal", frame.get("ordinal", 0))),
        "frame_id": _as_int(trace.get("frame_id", frame.get("frame_id", 0))),
        "invalid": bool(_get(trace, "frame_parsing", "invalid", default=frame.get("invalid", False))),
        "udp_gap_count": _as_int(_get(trace, "frame_parsing", "udp_gap_count", default=frame.get("udp_gap_count", 0))),
        "cfar_count": _as_int(_get(detection, "cfar", "candidate_count")),
        "angle_count": _as_int(_get(detection, "angle_validation", "passed_count")),
        "coarse_merge_count": _as_int(_get(detection, "candidate_merge_coarse", "after_count")),
        "body_count": _as_int(_get(detection, "body_center_refinement", "refined_count")),
        "final_merge_count": _as_int(_get(detection, "candidate_merge_final", "after_count")),
        "dbscan_input_count": _as_int(_get(detection, "dbscan", "input_count")),
        "dbscan_output_count": _as_int(_get(detection, "dbscan", "output_count")),
        "duplicate_before_count": _as_int(_get(detection, "duplicate_suppression", "before_count")),
        "duplicate_after_count": _as_int(_get(detection, "duplicate_suppression", "after_count")),
        "object_count_estimate": _as_int(_get(detection, "object_count_estimator", "estimated_count")),
        "tracker_input_count": _as_int(_get(trace, "tracker_input_filter", "tracker_input_count", default=frame.get("tracker_input_count", 0))),
        "tracker_measurement_count": _as_int(_get(tracker, "measurement_count")),
        "confirmed_count": _as_int(_get(display, "confirmed_count", default=len(frame.get("confirmed_tracks") or []))),
        "tentative_count": _as_int(_get(display, "tentative_count", default=len(frame.get("tentative_tracks") or []))),
        "display_count": _as_int(_get(display, "confirmed_count", default=len(frame.get("confirmed_tracks") or []))),
        "association_matched": _as_int(_get(tracker, "association", "matched_count")),
        "birth_count": len(lifecycle.get("births") or []),
        "suppressed_birth_count": len(lifecycle.get("suppressed_births") or []),
        "deleted_count": len(lifecycle.get("deleted_track_ids") or []),
        "rai_suspicious_count": _as_int(_get(trace, "rai_collapse_diagnostics", "suspicious_count")),
        "early_exit": str(_get(detection, "early_exit", default="") or ""),
    }


def _analyze(
    *,
    project_root: Path,
    session_id: str,
    mode: str,
    expected_max: int,
    min_separation_m: float,
    min_doppler_bins: int,
    min_score_ratio: float,
    jump_threshold_m: float,
) -> tuple[dict, list[dict], list[dict]]:
    cache_key = _stage_cache_key(session_id, mode)
    cache_dir = project_root / "lab_data" / "stage_cache" / cache_key
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    traces = _load_jsonl(cache_dir / "frame_trace.jsonl")
    frames = _load_jsonl(cache_dir / "frames.jsonl")
    frame_by_ordinal = {_as_int(row.get("ordinal")): row for row in frames}

    rows = []
    suspicious = []
    previous_by_track: dict[int, dict] = {}
    jump_events = []

    for trace in traces:
        ordinal = _as_int(trace.get("ordinal"))
        frame = frame_by_ordinal.get(ordinal, {})
        row = _frame_counts(trace, frame)
        candidates = _get(trace, "detection", "candidate_merge_final", "after_top", default=[]) or []
        evidence_count, evidence_candidates = _independent_candidate_count(
            candidates,
            min_separation_m=min_separation_m,
            min_doppler_bins=min_doppler_bins,
            min_score_ratio=min_score_ratio,
        )
        row["multi_evidence_count"] = int(evidence_count)
        if row.get("object_count_estimate", 0) > 0:
            row["multi_evidence_count"] = int(row["object_count_estimate"])
        row["multi_evidence"] = bool(evidence_count >= 2)
        if row.get("object_count_estimate", 0) > 0:
            row["multi_evidence"] = bool(row["object_count_estimate"] >= 2)
        row["evidence_candidate_preview"] = json.dumps(
            [
                {
                    "x_m": round(_as_float(item.get("x_m")), 3),
                    "y_m": round(_as_float(item.get("y_m")), 3),
                    "doppler_bin": _as_int(item.get("doppler_bin")),
                    "score": round(_candidate_strength(item), 4),
                }
                for item in evidence_candidates[:4]
            ],
            ensure_ascii=False,
        )

        reasons = []
        if row["multi_evidence"] and row["dbscan_output_count"] < 2:
            reasons.append("dbscan_collapse")
        if row["dbscan_output_count"] >= 2 and row["duplicate_after_count"] < 2:
            reasons.append("duplicate_collapse")
        if row["duplicate_after_count"] >= 2 and row["confirmed_count"] < 2:
            reasons.append("tracker_collapse")
        if row["confirmed_count"] >= 2 and row["display_count"] < 2:
            reasons.append("display_collapse")
        if expected_max > 0 and row["display_count"] > expected_max:
            reasons.append("over_expected_max")
        if row["rai_suspicious_count"] > 0:
            reasons.append("rai_suspicious")
        if row["early_exit"]:
            reasons.append(f"early_exit:{row['early_exit']}")

        for track in (frame.get("confirmed_tracks") or []):
            track_id = _as_int(track.get("track_id"), -1)
            if track_id < 0:
                continue
            x = _as_float(track.get("x_m"), math.nan)
            y = _as_float(track.get("y_m"), math.nan)
            previous = previous_by_track.get(track_id)
            if previous is not None and not math.isnan(x) and not math.isnan(y):
                step_m = math.hypot(x - previous["x_m"], y - previous["y_m"])
                if step_m > jump_threshold_m:
                    event = {
                        "ordinal": ordinal,
                        "frame_id": row["frame_id"],
                        "track_id": track_id,
                        "step_m": round(float(step_m), 4),
                        "from_x_m": round(previous["x_m"], 4),
                        "from_y_m": round(previous["y_m"], 4),
                        "to_x_m": round(x, 4),
                        "to_y_m": round(y, 4),
                    }
                    jump_events.append(event)
                    reasons.append("track_jump")
            if not math.isnan(x) and not math.isnan(y):
                previous_by_track[track_id] = {"x_m": x, "y_m": y}

        row["suspicious_reasons"] = ",".join(dict.fromkeys(reasons))
        rows.append(row)
        if reasons:
            suspicious.append(row)

    frame_count = len(rows)
    multi_frames = sum(1 for row in rows if row["multi_evidence"])
    summary = {
        "schema_version": 1,
        "session_id": session_id,
        "ablation_mode": str(mode or "baseline"),
        "cache_key": cache_key,
        "source_cache_dir": str(cache_dir),
        "capture_id": manifest.get("capture_id"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frame_count": frame_count,
        "expected_max": int(expected_max),
        "thresholds": {
            "min_separation_m": float(min_separation_m),
            "min_doppler_bins": int(min_doppler_bins),
            "min_score_ratio": float(min_score_ratio),
            "jump_threshold_m": float(jump_threshold_m),
        },
        "rates": {
            "multi_evidence_rate": _rate(multi_frames, frame_count),
            "dbscan_collapse_rate_on_multi_evidence": _rate(
                sum(1 for row in rows if row["multi_evidence"] and row["dbscan_output_count"] < 2),
                multi_frames,
            ),
            "tracker_collapse_rate_after_dbscan_multi": _rate(
                sum(1 for row in rows if row["dbscan_output_count"] >= 2 and row["confirmed_count"] < 2),
                sum(1 for row in rows if row["dbscan_output_count"] >= 2),
            ),
            "display_two_or_more_rate": _rate(sum(1 for row in rows if row["display_count"] >= 2), frame_count),
            "over_expected_max_rate": _rate(
                sum(1 for row in rows if expected_max > 0 and row["display_count"] > expected_max),
                frame_count,
            ),
            "track_jump_rate": _rate(len(jump_events), frame_count),
        },
        "distributions": {
            "final_merge_count": _count_distribution(rows, "final_merge_count"),
            "dbscan_output_count": _count_distribution(rows, "dbscan_output_count"),
            "confirmed_count": _count_distribution(rows, "confirmed_count"),
            "display_count": _count_distribution(rows, "display_count"),
            "multi_evidence_count": _count_distribution(rows, "multi_evidence_count"),
            "object_count_estimate": _count_distribution(rows, "object_count_estimate"),
        },
        "reason_counts": {},
        "jump_events": jump_events[:50],
    }
    for row in suspicious:
        for reason in row["suspicious_reasons"].split(","):
            if reason:
                summary["reason_counts"][reason] = summary["reason_counts"].get(reason, 0) + 1
    summary["reason_counts"] = dict(sorted(summary["reason_counts"].items(), key=lambda item: (-item[1], item[0])))
    return summary, rows, suspicious


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, summary: dict, suspicious: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary["rates"].items()
    )
    distributions = "\n".join(
        f"<h3>{html.escape(key)}</h3><pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
        for key, value in summary["distributions"].items()
    )
    reason_counts = html.escape(json.dumps(summary["reason_counts"], ensure_ascii=False, indent=2))
    suspicious_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key in [
                "ordinal",
                "frame_id",
                "object_count_estimate",
                "multi_evidence_count",
                "final_merge_count",
                "dbscan_output_count",
                "duplicate_after_count",
                "confirmed_count",
                "display_count",
                "suspicious_reasons",
            ]
        )
        + "</tr>"
        for row in suspicious[:80]
    )
    content = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Multi-target stage eval {html.escape(summary['session_id'])}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d7dde8; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #eef3f8; }}
    pre {{ background: #f6f8fb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    code {{ background: #f3f5f8; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Multi-target Stage Evaluation</h1>
  <p><strong>session:</strong> <code>{html.escape(summary['session_id'])}</code></p>
  <p><strong>ablation mode:</strong> <code>{html.escape(str(summary.get('ablation_mode', 'baseline')))}</code></p>
  <p><strong>capture:</strong> <code>{html.escape(str(summary.get('capture_id')))}</code></p>
  <p><strong>frame count:</strong> {summary['frame_count']}</p>
  <h2>Rates</h2>
  <table><tbody>{metric_rows}</tbody></table>
  <h2>Reason Counts</h2>
  <pre>{reason_counts}</pre>
  <h2>Distributions</h2>
  {distributions}
  <h2>Suspicious Frames</h2>
  <table>
    <thead><tr><th>ordinal</th><th>frame</th><th>object estimate</th><th>evidence</th><th>final merge</th><th>DBSCAN</th><th>dup after</th><th>confirmed</th><th>display</th><th>reasons</th></tr></thead>
    <tbody>{suspicious_rows}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-target count collapse using Stage Cache traces.")
    parser.add_argument("--session", required=True, help="Stage Cache session id.")
    parser.add_argument("--mode", default="baseline", help="Stage Cache ablation mode. Default: baseline.")
    parser.add_argument("--expected-max", type=int, default=2, help="Expected maximum visible object count.")
    parser.add_argument("--min-separation-m", type=float, default=0.65, help="Cartesian separation for independent candidate evidence.")
    parser.add_argument("--min-doppler-bins", type=int, default=7, help="Doppler separation for independent candidate evidence.")
    parser.add_argument("--min-score-ratio", type=float, default=0.05, help="Minimum candidate strength ratio to strongest candidate.")
    parser.add_argument("--jump-threshold-m", type=float, default=0.85, help="Per-track step threshold for jump flags.")
    parser.add_argument("--output-root", default="", help="Output root. Default: logs/diagnostics.")
    args = parser.parse_args()

    project_root = _repo_root()
    output_root = Path(args.output_root) if args.output_root else project_root / "logs" / "diagnostics"
    cache_key = _stage_cache_key(str(args.session), str(args.mode))
    cache_dir = project_root / "lab_data" / "stage_cache" / cache_key
    if not (cache_dir / "frame_trace.jsonl").exists():
        raise SystemExit(
            "Stage Cache not found for "
            f"{args.session} mode={args.mode}. "
            f"Run: python -B -m tools.lab.stage_cache --session {args.session} --mode {args.mode} --force"
        )

    summary, rows, suspicious = _analyze(
        project_root=project_root,
        session_id=str(args.session),
        mode=str(args.mode),
        expected_max=int(args.expected_max),
        min_separation_m=float(args.min_separation_m),
        min_doppler_bins=int(args.min_doppler_bins),
        min_score_ratio=float(args.min_score_ratio),
        jump_threshold_m=float(args.jump_threshold_m),
    )
    output_dir = output_root / f"multitarget_stage_eval_{_now_tag()}_{cache_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    frames_path = output_dir / "frame_counts.csv"
    suspicious_path = output_dir / "suspicious_frames.csv"
    report_path = output_dir / "report.html"
    summary["output_dir"] = str(output_dir)
    summary["frames_csv"] = str(frames_path)
    summary["suspicious_csv"] = str(suspicious_path)
    summary["report_html"] = str(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(frames_path, rows)
    _write_csv(suspicious_path, suspicious)
    _write_html(report_path, summary, suspicious)

    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Suspicious frames: {suspicious_path}")
    print(json.dumps({"rates": summary["rates"], "reason_counts": summary["reason_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
