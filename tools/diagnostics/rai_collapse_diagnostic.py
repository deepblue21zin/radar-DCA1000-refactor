from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path

import numpy as np

from tools.lab import registry
from tools.lab.stage_cache import (
    _build_runtime_components,
    _merge_runtime_summary,
    _resolve_capture_dir,
)
from tools.runtime_core import DSP
from tools.runtime_core.detection import detect_targets
from tools.runtime_core.radar_runtime import (
    collapse_motion_rai,
    frame_to_radar_cube,
    integrate_rdi_channels,
    remove_static_clutter,
)
from tools.runtime_core.real_time_process import (
    iter_raw_capture_frame_packets,
    load_raw_capture,
    project_range_angle_cube,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_session_context(project_root: Path, session_id: str) -> tuple[Path, dict]:
    run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        registry.refresh_registry(project_root)
        run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        raise FileNotFoundError(f"Run session not found in registry: {session_id}")

    capture_dir = _resolve_capture_dir(project_root, run_detail)
    capture_manifest, _, _ = load_raw_capture(capture_dir)
    runtime_summary = _merge_runtime_summary(run_detail, capture_manifest)
    return capture_dir, runtime_summary


def _load_capture_context(project_root: Path, capture_value: str) -> tuple[Path, dict]:
    capture_dir = Path(capture_value)
    if not capture_dir.is_absolute():
        capture_dir = project_root / capture_dir
    capture_dir = capture_dir.resolve()
    capture_manifest, _, _ = load_raw_capture(capture_dir)
    runtime_summary = capture_manifest.get("runtime_summary") or {}
    if not runtime_summary:
        raise ValueError(f"Capture manifest does not contain runtime_summary: {capture_dir}")
    return capture_dir, runtime_summary


def _round_or_none(value, digits=4):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _wrap_angle_delta_deg(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    delta = (float(left) - float(right) + 180.0) % 360.0 - 180.0
    return float(delta)


def _sign_label(value: float | None, deadband=0.05) -> str:
    if value is None:
        return "none"
    if float(value) > deadband:
        return "right"
    if float(value) < -deadband:
        return "left"
    return "center"


def _angle_mask_for_range(runtime_config, range_m: float, detection_region) -> np.ndarray:
    angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
    x_axis = range_m * np.sin(angle_axis)
    y_axis = range_m * np.cos(angle_axis)
    return (
        (np.abs(x_axis) <= float(detection_region.lateral_limit_m))
        & (y_axis >= float(detection_region.min_forward_m))
        & (y_axis <= float(detection_region.forward_limit_m))
    )


def _peak_from_profile(profile, runtime_config, range_m: float, angle_mask) -> dict:
    values = np.asarray(profile, dtype=np.float64)
    mask = np.asarray(angle_mask, dtype=bool)
    if values.size == 0:
        return {}
    if mask.shape != values.shape or not np.any(mask):
        mask = np.ones(values.shape, dtype=bool)

    masked = np.where(mask, values, 0.0)
    peak_bin = int(np.argmax(masked))
    peak_power = float(masked[peak_bin])
    angle_rad = float(np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)[peak_bin])
    x_m = float(range_m * math.sin(angle_rad))
    y_m = float(range_m * math.cos(angle_rad))
    return {
        "angle_bin": peak_bin,
        "angle_deg": math.degrees(angle_rad),
        "power": peak_power,
        "x_m": x_m,
        "y_m": y_m,
        "side": _sign_label(x_m),
    }


def _fallback_rdi_candidates(rdi, runtime_config, min_range_bin, max_range_bin, limit: int) -> list[dict]:
    rdi_roi = np.asarray(rdi[min_range_bin:max_range_bin], dtype=np.float64)
    if rdi_roi.size == 0:
        return []
    work = np.array(rdi_roi, copy=True)
    center_bin = int(runtime_config.doppler_fft_size // 2)
    guard_bins = int(runtime_config.doppler_guard_bins)
    lower = max(center_bin - guard_bins, 0)
    upper = min(center_bin + guard_bins + 1, runtime_config.doppler_fft_size)
    work[:, lower:upper] = 0.0
    work = np.maximum(work - np.median(work, axis=1, keepdims=True), 0.0)
    flat_order = np.argsort(np.square(work).ravel())[::-1]
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for flat_index in flat_order:
        if len(rows) >= limit:
            break
        range_rel, doppler_bin = np.unravel_index(int(flat_index), work.shape)
        if float(work[range_rel, doppler_bin]) <= 0.0:
            break
        key = (int(range_rel), int(doppler_bin))
        if key in seen:
            continue
        seen.add(key)
        range_bin = int(range_rel + min_range_bin)
        rows.append(
            {
                "range_bin": range_bin,
                "doppler_bin": int(doppler_bin),
                "range_m": float(runtime_config.range_axis_m[range_bin]),
                "power": float(np.square(work[range_rel, doppler_bin])),
                "source": "fallback_rdi_power",
            }
        )
    return rows


def _candidate_rows_from_trace(trace: dict, runtime_config, limit: int) -> list[dict]:
    candidates = (((trace or {}).get("cfar") or {}).get("top_candidates") or [])[:limit]
    rows = []
    for candidate in candidates:
        range_bin = int(candidate.get("range_bin"))
        rows.append(
            {
                "range_bin": range_bin,
                "doppler_bin": int(candidate.get("doppler_bin")),
                "range_m": float(candidate.get("range_m", runtime_config.range_axis_m[range_bin])),
                "power": float(candidate.get("power", 0.0)),
                "source": "cfar_top",
            }
        )
    return rows


def _serialize_detection(detection) -> dict:
    return {
        "range_bin": int(detection.range_bin),
        "doppler_bin": int(detection.doppler_bin),
        "angle_deg": round(float(detection.angle_deg), 4),
        "x_m": round(float(detection.x_m), 4),
        "y_m": round(float(detection.y_m), 4),
        "score": round(float(detection.score), 4),
    }


def _diagnose_frame(raw_frame, components: dict, *, top_rdi: int, angle_delta_threshold: float, x_delta_threshold: float):
    runtime_config = components["runtime_config"]
    detection_region = components["detection_region"]

    radar_cube = frame_to_radar_cube(raw_frame.iq, runtime_config)
    if runtime_config.remove_static:
        radar_cube = remove_static_clutter(radar_cube)

    shared_fft = DSP.shared_range_doppler_fft(
        radar_cube,
        padding_size=[
            runtime_config.doppler_fft_size,
            runtime_config.range_fft_size,
        ],
    )
    rdi_cube = DSP.range_doppler_from_fft(shared_fft, mode=1)
    rdi = integrate_rdi_channels(rdi_cube)

    rai_cube = project_range_angle_cube(shared_fft, runtime_config)
    rai_current = collapse_motion_rai(rai_cube, guard_bins=runtime_config.doppler_guard_bins)
    rai_cube_legacy_unshifted = np.fft.ifftshift(rai_cube, axes=0)
    rai_legacy_collapse = collapse_motion_rai(
        rai_cube_legacy_unshifted,
        guard_bins=runtime_config.doppler_guard_bins,
    )

    detection_trace: dict = {}
    detections = detect_targets(
        rdi,
        rai_current,
        runtime_config,
        components["min_range_bin"],
        components["max_range_bin"],
        detection_region,
        rai_cube=rai_cube,
        trace=detection_trace,
        **dict(components["detection_params"] or {}),
    )

    rdi_candidates = _candidate_rows_from_trace(detection_trace, runtime_config, top_rdi)
    if not rdi_candidates:
        rdi_candidates = _fallback_rdi_candidates(
            rdi,
            runtime_config,
            components["min_range_bin"],
            components["max_range_bin"],
            top_rdi,
        )

    top_detection = _serialize_detection(detections[0]) if detections else {}
    rows = []
    for rank, candidate in enumerate(rdi_candidates, start=1):
        range_bin = int(candidate["range_bin"])
        doppler_bin = int(candidate["doppler_bin"])
        if range_bin < 0 or range_bin >= rai_current.shape[0]:
            continue
        if doppler_bin < 0 or doppler_bin >= rai_cube.shape[0]:
            continue

        range_m = float(candidate["range_m"])
        angle_mask = _angle_mask_for_range(runtime_config, range_m, detection_region)
        slice_peak = _peak_from_profile(
            rai_cube[doppler_bin, range_bin, :],
            runtime_config,
            range_m,
            angle_mask,
        )
        current_peak = _peak_from_profile(
            rai_current[range_bin, :],
            runtime_config,
            range_m,
            angle_mask,
        )
        legacy_peak = _peak_from_profile(
            rai_legacy_collapse[range_bin, :],
            runtime_config,
            range_m,
            angle_mask,
        )
        if not slice_peak or not current_peak or not legacy_peak:
            continue

        current_delta = _wrap_angle_delta_deg(current_peak["angle_deg"], slice_peak["angle_deg"])
        legacy_delta = _wrap_angle_delta_deg(legacy_peak["angle_deg"], slice_peak["angle_deg"])
        current_x_delta = float(current_peak["x_m"] - slice_peak["x_m"])
        legacy_x_delta = float(legacy_peak["x_m"] - slice_peak["x_m"])
        current_sign_changed = slice_peak["side"] != "center" and current_peak["side"] not in {
            "center",
            slice_peak["side"],
        }
        legacy_sign_changed = slice_peak["side"] != "center" and legacy_peak["side"] not in {
            "center",
            slice_peak["side"],
        }
        current_suspicious = (
            abs(float(current_delta)) >= float(angle_delta_threshold)
            or abs(current_x_delta) >= float(x_delta_threshold)
            or current_sign_changed
        )
        legacy_suspicious = (
            abs(float(legacy_delta)) >= float(angle_delta_threshold)
            or abs(legacy_x_delta) >= float(x_delta_threshold)
            or legacy_sign_changed
        )
        if current_suspicious and not legacy_suspicious:
            verdict = "current_collapse_suspect"
        elif current_suspicious and legacy_suspicious:
            verdict = "motion_collapse_or_frontend_ghost_suspect"
        elif not current_suspicious and legacy_suspicious:
            verdict = "legacy_unshifted_differs_only"
        else:
            verdict = "slice_and_collapse_agree"

        row = {
            "ordinal": None,
            "frame_id": int(raw_frame.frame_id),
            "capture_ts": _round_or_none(raw_frame.capture_ts, 6),
            "candidate_rank": int(rank),
            "candidate_source": candidate.get("source", ""),
            "range_bin": range_bin,
            "range_m": _round_or_none(range_m),
            "doppler_bin": doppler_bin,
            "rdi_power": _round_or_none(candidate.get("power")),
            "slice_angle_deg": _round_or_none(slice_peak["angle_deg"], 3),
            "slice_power": _round_or_none(slice_peak["power"]),
            "slice_x_m": _round_or_none(slice_peak["x_m"]),
            "slice_y_m": _round_or_none(slice_peak["y_m"]),
            "slice_side": slice_peak["side"],
            "current_collapsed_angle_deg": _round_or_none(current_peak["angle_deg"], 3),
            "current_collapsed_power": _round_or_none(current_peak["power"]),
            "current_collapsed_x_m": _round_or_none(current_peak["x_m"]),
            "current_collapsed_side": current_peak["side"],
            "current_minus_slice_angle_deg": _round_or_none(current_delta, 3),
            "current_minus_slice_x_m": _round_or_none(current_x_delta),
            "current_sign_changed": bool(current_sign_changed),
            "current_suspicious": bool(current_suspicious),
            "legacy_unshifted_angle_deg": _round_or_none(legacy_peak["angle_deg"], 3),
            "legacy_unshifted_power": _round_or_none(legacy_peak["power"]),
            "legacy_unshifted_x_m": _round_or_none(legacy_peak["x_m"]),
            "legacy_unshifted_side": legacy_peak["side"],
            "legacy_minus_slice_angle_deg": _round_or_none(legacy_delta, 3),
            "legacy_minus_slice_x_m": _round_or_none(legacy_x_delta),
            "legacy_sign_changed": bool(legacy_sign_changed),
            "legacy_suspicious": bool(legacy_suspicious),
            "detection_count": int(len(detections)),
            "top_detection_angle_deg": top_detection.get("angle_deg"),
            "top_detection_x_m": top_detection.get("x_m"),
            "top_detection_y_m": top_detection.get("y_m"),
            "top_detection_doppler_bin": top_detection.get("doppler_bin"),
            "verdict": verdict,
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "frame_id",
        "candidate_rank",
        "verdict",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_html_report(path: Path, summary: dict, rows: list[dict]) -> None:
    suspicious = [row for row in rows if row.get("current_suspicious") or row.get("shifted_suspicious")]
    preview_rows = suspicious[:120] if suspicious else rows[:120]
    columns = [
        "frame_id",
        "candidate_rank",
        "range_m",
        "doppler_bin",
        "slice_angle_deg",
        "current_collapsed_angle_deg",
        "legacy_unshifted_angle_deg",
        "current_minus_slice_angle_deg",
        "legacy_minus_slice_angle_deg",
        "slice_side",
        "current_collapsed_side",
        "legacy_unshifted_side",
        "verdict",
    ]
    table_rows = []
    for row in preview_rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        table_rows.append(f"<tr>{cells}</tr>")
    table_header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    metric_cards = "".join(
        f"<div class='card'><div class='label'>{html.escape(str(key))}</div>"
        f"<div class='value'>{html.escape(str(value))}</div></div>"
        for key, value in summary.get("metrics", {}).items()
    )
    path.write_text(
        f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>RAI Collapse Diagnostic</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #15202b; }}
    h1 {{ margin-bottom: 4px; }}
    code {{ background: #f4f6f8; padding: 2px 5px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; background: #fbfdff; }}
    .label {{ color: #657786; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6ecf1; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    th {{ position: sticky; top: 0; background: white; }}
    .hint {{ background: #fff8e1; border-left: 4px solid #f4b400; padding: 10px 12px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>RAI Collapse Diagnostic</h1>
  <p>RDI 후보의 Doppler slice angle과 현재 collapsed RAI angle을 비교한 오프라인 진단 결과입니다.</p>
  <div class="hint">
    <b>해석:</b> <code>slice_angle_deg</code>는 RDI 후보의 Doppler bin에서 직접 본 angle이고,
    <code>current_collapsed_angle_deg</code>는 현재 파이프라인의 2D RAI collapse 후 선택되는 angle입니다.
    <code>legacy_unshifted_angle_deg</code>는 Doppler 축 정렬 전 가정으로 collapse했을 때의 비교값입니다.
    slice/current의 좌우 부호가 바뀌거나 angle 차이가 크면 RAI collapse 또는 전단 ghost 문제를 의심합니다.
  </div>
  <div class="grid">{metric_cards}</div>
  <p><b>capture:</b> <code>{html.escape(str(summary.get("capture_dir", "")))}</code></p>
  <p><b>cfg:</b> <code>{html.escape(str(summary.get("cfg_path", "")))}</code></p>
  <p><b>angle projection:</b> <code>{html.escape(str(summary.get("angle_projection", "")))}</code></p>
  <h2>Suspicious Preview</h2>
  <table><thead><tr>{table_header}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body>
</html>
""",
        encoding="utf-8",
    )


def run_diagnostic(
    *,
    project_root: Path,
    session_id: str | None,
    capture: str | None,
    output_dir: Path | None,
    start: int,
    limit: int | None,
    top_rdi: int,
    angle_delta_threshold: float,
    x_delta_threshold: float,
) -> dict:
    if bool(session_id) == bool(capture):
        raise ValueError("Provide exactly one of --session or --capture.")

    if session_id:
        capture_dir, runtime_summary = _load_session_context(project_root, session_id)
        label = session_id
    else:
        capture_dir, runtime_summary = _load_capture_context(project_root, str(capture))
        label = capture_dir.name

    components = _build_runtime_components(project_root, runtime_summary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = project_root / "logs" / "diagnostics" / f"rai_collapse_{timestamp}_{label}"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    processed_frames = 0
    for ordinal, raw_frame in enumerate(iter_raw_capture_frame_packets(capture_dir)):
        if ordinal < int(start):
            continue
        if limit is not None and processed_frames >= int(limit):
            break
        frame_rows = _diagnose_frame(
            raw_frame,
            components,
            top_rdi=int(top_rdi),
            angle_delta_threshold=float(angle_delta_threshold),
            x_delta_threshold=float(x_delta_threshold),
        )
        for row in frame_rows:
            row["ordinal"] = int(ordinal)
        rows.extend(frame_rows)
        processed_frames += 1

    current_suspicious = sum(1 for row in rows if bool(row.get("current_suspicious")))
    legacy_suspicious = sum(1 for row in rows if bool(row.get("legacy_suspicious")))
    current_only = sum(
        1
        for row in rows
        if bool(row.get("current_suspicious")) and not bool(row.get("legacy_suspicious"))
    )
    both_suspicious = sum(
        1
        for row in rows
        if bool(row.get("current_suspicious")) and bool(row.get("legacy_suspicious"))
    )
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "capture_dir": str(capture_dir),
        "output_dir": str(output_dir),
        "cfg_path": str(components["cfg_path"]),
        "angle_projection": components["runtime_config"].angle_projection,
        "angle_phase_sign": components["runtime_config"].angle_phase_sign,
        "doppler_guard_bins": components["runtime_config"].doppler_guard_bins,
        "top_rdi": int(top_rdi),
        "start": int(start),
        "limit": limit,
        "metrics": {
            "frames_processed": int(processed_frames),
            "rows": int(len(rows)),
            "current_suspicious_rows": int(current_suspicious),
            "legacy_unshifted_suspicious_rows": int(legacy_suspicious),
            "current_only_rows": int(current_only),
            "both_suspicious_rows": int(both_suspicious),
        },
        "verdict_counts": verdict_counts,
    }

    csv_path = output_dir / "rai_collapse_diagnostic.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.html"
    _write_csv(csv_path, rows)
    _write_json(summary_path, summary)
    _write_html_report(report_path, summary, rows)

    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Doppler-slice angle peaks against collapsed RAI angle peaks "
            "for a saved raw capture or live/replay session."
        )
    )
    parser.add_argument("--session", help="logs/live_motion_viewer session id with linked raw capture.")
    parser.add_argument("--capture", help="Raw capture directory, e.g. logs/raw/20260508_193458.")
    parser.add_argument("--out", help="Output directory. Defaults to logs/diagnostics/rai_collapse_<time>_<id>.")
    parser.add_argument("--start", type=int, default=0, help="Raw frame ordinal to start from.")
    parser.add_argument("--limit", type=int, default=80, help="Maximum frames to process. Use 0 for all frames.")
    parser.add_argument("--top-rdi", type=int, default=4, help="RDI candidates to compare per frame.")
    parser.add_argument("--angle-delta-deg", type=float, default=15.0, help="Suspicious angle delta threshold.")
    parser.add_argument("--x-delta-m", type=float, default=0.35, help="Suspicious x displacement threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_diagnostic(
        project_root=PROJECT_ROOT,
        session_id=args.session,
        capture=args.capture,
        output_dir=Path(args.out) if args.out else None,
        start=max(0, int(args.start)),
        limit=None if int(args.limit) <= 0 else int(args.limit),
        top_rdi=max(1, int(args.top_rdi)),
        angle_delta_threshold=float(args.angle_delta_deg),
        x_delta_threshold=float(args.x_delta_m),
    )
    print(f"RAI collapse diagnostic report: {summary['report_path']}")
    print(f"CSV: {summary['csv_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("Metrics:", json.dumps(summary["metrics"], ensure_ascii=False))
    print("Verdicts:", json.dumps(summary["verdict_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
