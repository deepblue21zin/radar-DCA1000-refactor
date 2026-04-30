from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _point_from_item(item: dict[str, Any], frame_id: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    x_m = _as_float(item.get("x_m"))
    y_m = _as_float(item.get("y_m"))
    if x_m is None or y_m is None:
        return None
    return {
        "frame_id": int(frame_id),
        "x_m": x_m,
        "y_m": y_m,
        "score": _as_float(item.get("score")) or 0.0,
        "confidence": _as_float(item.get("confidence")) or 0.0,
        "rdi_peak": _as_float(item.get("rdi_peak")) or 0.0,
        "rai_peak": _as_float(item.get("rai_peak")) or 0.0,
        "hits": int(item.get("hits") or 0),
        "age": int(item.get("age") or 0),
        "is_primary": bool(item.get("is_primary")),
        "track_id": item.get("track_id"),
    }


def _rank_detection(point: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(point.get("score") or 0.0),
        float(point.get("rai_peak") or 0.0),
        float(point.get("rdi_peak") or 0.0),
    )


def _rank_track(point: dict[str, Any]) -> tuple[int, float, float, int, int]:
    return (
        1 if point.get("is_primary") else 0,
        float(point.get("confidence") or 0.0),
        float(point.get("score") or 0.0),
        int(point.get("hits") or 0),
        int(point.get("age") or 0),
    )


def _collect_lead_points(
    records: list[dict[str, Any]],
    keys: list[str],
    *,
    rank_kind: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    ranker = _rank_detection if rank_kind == "detection" else _rank_track
    for record in records:
        frame_id = int(record.get("frame_id", record.get("frame_index", 0)) or 0)
        for key in keys:
            items = record.get(key) or []
            if not isinstance(items, list) or not items:
                continue
            candidates = [
                point
                for item in items
                if (point := _point_from_item(item, frame_id)) is not None
            ]
            if not candidates:
                continue
            lead = max(candidates, key=ranker)
            lead["source_key"] = key
            points.append(lead)
            break
    points.sort(key=lambda point: int(point["frame_id"]))
    return points


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _path_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    if len(points) < 3:
        return {
            "point_count": len(points),
            "x_span_m": None,
            "y_span_m": None,
            "major_span_m": None,
            "minor_span_m": None,
            "width_ratio": None,
            "line_residual_rms_m": None,
            "line_residual_p95_m": None,
            "step_p50_m": None,
            "step_p95_m": None,
            "max_step_m": None,
            "step_jump_ratio": None,
        }

    xs = [float(point["x_m"]) for point in points]
    ys = [float(point["y_m"]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered = [(x - mean_x, y - mean_y) for x, y in zip(xs, ys)]

    sxx = sum(x * x for x, _ in centered) / len(centered)
    syy = sum(y * y for _, y in centered) / len(centered)
    sxy = sum(x * y for x, y in centered) / len(centered)
    trace = sxx + syy
    determinant = (sxx * syy) - (sxy * sxy)
    discriminant = max((trace * trace / 4.0) - determinant, 0.0)
    root = math.sqrt(discriminant)
    major_var = max(trace / 2.0 + root, 0.0)
    minor_var = max(trace / 2.0 - root, 0.0)

    if abs(sxy) > 1e-12 or abs(major_var - sxx) > 1e-12:
        vx = sxy
        vy = major_var - sxx
        norm = math.hypot(vx, vy)
        if norm <= 1e-12:
            vx, vy = 1.0, 0.0
        else:
            vx, vy = vx / norm, vy / norm
    else:
        vx, vy = 1.0, 0.0

    nx, ny = -vy, vx
    along = [(x * vx) + (y * vy) for x, y in centered]
    across = [(x * nx) + (y * ny) for x, y in centered]
    residuals = [abs(value) for value in across]
    steps = [
        math.hypot(
            float(current["x_m"]) - float(previous["x_m"]),
            float(current["y_m"]) - float(previous["y_m"]),
        )
        for previous, current in zip(points, points[1:])
        if int(current["frame_id"]) - int(previous["frame_id"]) <= 2
    ]
    step_p50 = _quantile(steps, 0.50)
    jump_threshold = None if step_p50 is None else max(0.35, float(step_p50) * 3.0)
    step_jump_ratio = None
    if steps and jump_threshold is not None:
        step_jump_ratio = sum(1 for step in steps if step > jump_threshold) / len(steps)

    major_span = max(along) - min(along) if along else None
    minor_span = max(across) - min(across) if across else None
    width_ratio = None
    if major_span is not None and minor_span is not None and major_span > 1e-9:
        width_ratio = minor_span / major_span

    residual_rms = None
    if residuals:
        residual_rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))

    return {
        "point_count": len(points),
        "x_span_m": _round(max(xs) - min(xs)),
        "y_span_m": _round(max(ys) - min(ys)),
        "major_span_m": _round(major_span),
        "minor_span_m": _round(minor_span),
        "width_ratio": _round(width_ratio),
        "line_residual_rms_m": _round(residual_rms),
        "line_residual_p95_m": _round(_quantile(residuals, 0.95)),
        "step_p50_m": _round(step_p50),
        "step_p95_m": _round(_quantile(steps, 0.95)),
        "max_step_m": _round(max(steps) if steps else None),
        "step_jump_ratio": _round(step_jump_ratio),
    }


def _delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return round(float(after) - float(before), 4)


def _ratio(after: float | None, before: float | None) -> float | None:
    if after is None or before in (None, 0):
        return None
    return round(float(after) / float(before), 4)


def _path_comparison(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        "x_span_delta_m": _delta(after.get("x_span_m"), before.get("x_span_m")),
        "x_span_ratio": _ratio(after.get("x_span_m"), before.get("x_span_m")),
        "y_span_delta_m": _delta(after.get("y_span_m"), before.get("y_span_m")),
        "y_span_ratio": _ratio(after.get("y_span_m"), before.get("y_span_m")),
        "major_span_delta_m": _delta(after.get("major_span_m"), before.get("major_span_m")),
        "major_span_ratio": _ratio(after.get("major_span_m"), before.get("major_span_m")),
        "minor_span_delta_m": _delta(after.get("minor_span_m"), before.get("minor_span_m")),
        "minor_span_ratio": _ratio(after.get("minor_span_m"), before.get("minor_span_m")),
        "line_residual_rms_delta_m": _delta(
            after.get("line_residual_rms_m"),
            before.get("line_residual_rms_m"),
        ),
        "line_residual_rms_ratio": _ratio(
            after.get("line_residual_rms_m"),
            before.get("line_residual_rms_m"),
        ),
        "width_ratio_delta": _delta(after.get("width_ratio"), before.get("width_ratio")),
        "width_ratio_ratio": _ratio(after.get("width_ratio"), before.get("width_ratio")),
        "step_p95_delta_m": _delta(after.get("step_p95_m"), before.get("step_p95_m")),
        "step_p95_ratio": _ratio(after.get("step_p95_m"), before.get("step_p95_m")),
        "max_step_delta_m": _delta(after.get("max_step_m"), before.get("max_step_m")),
        "max_step_ratio": _ratio(after.get("max_step_m"), before.get("max_step_m")),
    }


def _paired_distance_metrics(
    raw_points: list[dict[str, Any]],
    output_points: list[dict[str, Any]],
    *,
    frame_tolerance: int = 2,
) -> dict[str, Any]:
    if not raw_points or not output_points:
        return {
            "paired_count": 0,
            "paired_output_ratio": None,
            "paired_raw_ratio": None,
            "distance_mean_m": None,
            "distance_rms_m": None,
            "distance_p95_m": None,
            "distance_max_m": None,
        }

    raw_by_frame = {int(point["frame_id"]): point for point in raw_points}
    distances: list[float] = []
    for output in output_points:
        frame_id = int(output["frame_id"])
        best_raw = None
        best_gap = None
        for candidate_frame in range(frame_id - frame_tolerance, frame_id + frame_tolerance + 1):
            raw = raw_by_frame.get(candidate_frame)
            if raw is None:
                continue
            gap = abs(candidate_frame - frame_id)
            if best_gap is None or gap < best_gap:
                best_raw = raw
                best_gap = gap
        if best_raw is None:
            continue
        distances.append(
            math.hypot(
                float(output["x_m"]) - float(best_raw["x_m"]),
                float(output["y_m"]) - float(best_raw["y_m"]),
            )
        )

    rms = None
    if distances:
        rms = math.sqrt(sum(distance * distance for distance in distances) / len(distances))

    return {
        "paired_count": len(distances),
        "paired_output_ratio": _round(len(distances) / len(output_points), digits=4),
        "paired_raw_ratio": _round(len(distances) / len(raw_points), digits=4),
        "distance_mean_m": _round(sum(distances) / len(distances) if distances else None),
        "distance_rms_m": _round(rms),
        "distance_p95_m": _round(_quantile(distances, 0.95)),
        "distance_max_m": _round(max(distances) if distances else None),
    }


def _raw_quality(raw: dict[str, Any]) -> dict[str, Any]:
    raw_step_p95 = raw.get("step_p95_m")
    raw_step_p50 = raw.get("step_p50_m")
    raw_jump_ratio = raw.get("step_jump_ratio")

    dynamic_step_limit = 0.35
    if raw_step_p50 is not None:
        dynamic_step_limit = max(dynamic_step_limit, float(raw_step_p50) * 3.5)

    is_good = (
        raw.get("point_count", 0) >= 8
        and raw_step_p95 is not None
        and raw_jump_ratio is not None
        and float(raw_step_p95) <= dynamic_step_limit
        and float(raw_jump_ratio) <= 0.08
    )
    is_jumpy = (
        raw.get("point_count", 0) >= 8
        and raw_step_p95 is not None
        and raw_jump_ratio is not None
        and (
            float(raw_step_p95) >= max(0.45, float(raw_step_p50 or 0.0) * 4.0)
            or float(raw_jump_ratio) >= 0.12
        )
    )
    return {
        "raw_quality_good": bool(is_good),
        "raw_quality_jumpy": bool(is_jumpy),
        "dynamic_step_p95_limit_m": _round(dynamic_step_limit),
    }


def _tracking_output_policy(
    tracking: dict[str, Any],
    output: dict[str, Any],
    output_vs_tracking: dict[str, Any],
) -> dict[str, Any]:
    track_point_count = int(tracking.get("point_count") or 0)
    output_point_count = int(output.get("point_count") or 0)
    track_x_span = tracking.get("x_span_m")
    track_major_span = tracking.get("major_span_m")
    track_step_p95 = tracking.get("step_p95_m")
    output_max_step = output.get("max_step_m")

    x_span_meaningful = (
        track_point_count >= 8
        and track_x_span is not None
        and float(track_x_span) >= 0.35
    )
    x_span_ratio = output_vs_tracking.get("x_span_ratio")
    x_span_preserved = True
    if x_span_meaningful:
        x_span_preserved = x_span_ratio is not None and float(x_span_ratio) >= 0.50

    major_span_meaningful = (
        track_point_count >= 8
        and track_major_span is not None
        and float(track_major_span) >= 0.70
    )
    major_span_ratio = output_vs_tracking.get("major_span_ratio")
    major_span_preserved = True
    if major_span_meaningful:
        major_span_preserved = major_span_ratio is not None and float(major_span_ratio) >= 0.75

    max_step_limit = None
    output_step_bound_pass = True
    if output_point_count >= 8 and track_step_p95 is not None:
        max_step_limit = max(0.85, float(track_step_p95) * 4.0)
        output_step_bound_pass = (
            output_max_step is not None and float(output_max_step) <= max_step_limit
        )

    return {
        "tracking_reference_point_count": track_point_count,
        "tracking_x_span_meaningful": bool(x_span_meaningful),
        "output_preserves_tracking_x_span": bool(x_span_preserved),
        "output_preserves_tracking_major_span": bool(major_span_preserved),
        "output_step_bound_limit_m": _round(max_step_limit),
        "output_step_bound_pass": bool(output_step_bound_pass),
        "output_preserves_tracking_shape": bool(
            x_span_preserved and major_span_preserved and output_step_bound_pass
        ),
    }


def _policy(
    raw: dict[str, Any],
    output: dict[str, Any],
    fidelity: dict[str, Any],
    *,
    tracking: dict[str, Any] | None = None,
    output_vs_tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = _raw_quality(raw)
    raw_step_p95 = raw.get("step_p95_m")
    output_step_p95 = output.get("step_p95_m")

    output_preserves_good_raw = True
    if quality["raw_quality_good"]:
        output_preserves_good_raw = (
            fidelity.get("paired_count", 0) >= 8
            and fidelity.get("distance_rms_m") is not None
            and fidelity.get("distance_p95_m") is not None
            and float(fidelity["distance_rms_m"]) <= 0.18
            and float(fidelity["distance_p95_m"]) <= 0.35
            and (fidelity.get("paired_output_ratio") or 0.0) >= 0.70
        )

    output_smooths_jumpy_raw = True
    if quality["raw_quality_jumpy"]:
        output_smooths_jumpy_raw = (
            raw_step_p95 is not None
            and output_step_p95 is not None
            and float(output_step_p95) <= float(raw_step_p95) * 0.9
            and (
                fidelity.get("distance_p95_m") is None
                or float(fidelity["distance_p95_m"]) <= max(0.60, float(raw_step_p95))
            )
        )

    tracking_policy = _tracking_output_policy(
        tracking or {},
        output,
        output_vs_tracking or {},
    )

    return {
        **quality,
        **tracking_policy,
        "output_preserves_good_raw": bool(output_preserves_good_raw),
        "output_smooths_jumpy_raw": bool(output_smooths_jumpy_raw),
        "overall_pass": bool(
            output_preserves_good_raw
            and output_smooths_jumpy_raw
            and tracking_policy["output_preserves_tracking_shape"]
        ),
        "rule": (
            "If raw-like detections are already good, final output should stay close to the measured trajectory. "
            "If raw-like detections are jumpy, final output should reduce step outliers while avoiding large drift from the measured trajectory. "
            "When a stable tracking trajectory exists, render/display output should not collapse its lateral span or introduce large display jumps."
        ),
    }


def _straightness_note(raw: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    raw_residual = raw.get("line_residual_rms_m")
    output_residual = output.get("line_residual_rms_m")
    raw_width = raw.get("width_ratio")
    output_width = output.get("width_ratio")
    raw_is_straightish = (
        raw.get("point_count", 0) >= 8
        and raw_residual is not None
        and raw_width is not None
        and float(raw_residual) <= 0.12
        and float(raw_width) <= 0.28
    )
    output_preserves_straightish_shape = True
    if raw_is_straightish:
        output_preserves_straightish_shape = (
            output_residual is not None
            and output_width is not None
            and float(output_residual) <= max(0.16, float(raw_residual) * 1.35)
            and float(output_width) <= max(0.34, float(raw_width) * 1.35)
        )
    return {
        "raw_is_straightish": bool(raw_is_straightish),
        "output_preserves_straightish_shape": bool(output_preserves_straightish_shape),
        "note": "This is diagnostic only. Passing the harness is based on measured-trajectory fidelity, not forced straightness.",
    }


def build_path_shape_metrics(session_dir: Path) -> dict[str, Any]:
    session_dir = Path(session_dir)
    processed_records = _load_jsonl(session_dir / "processed_frames.jsonl")
    render_records = _load_jsonl(session_dir / "render_frames.jsonl")

    raw_like_points = _collect_lead_points(
        processed_records,
        ["detections"],
        rank_kind="detection",
    )
    tracking_points = _collect_lead_points(
        processed_records,
        ["confirmed_tracks", "tentative_tracks"],
        rank_kind="track",
    )
    output_points = _collect_lead_points(
        render_records,
        ["display_tracks", "tentative_display_tracks"],
        rank_kind="track",
    )
    output_source = "render"
    if len(output_points) < 3:
        output_points = _collect_lead_points(
            processed_records,
            ["confirmed_tracks", "tentative_tracks"],
            rank_kind="track",
        )
        output_source = "processed"

    raw_metrics = _path_metrics(raw_like_points)
    tracking_metrics = _path_metrics(tracking_points)
    output_metrics = _path_metrics(output_points)
    fidelity = _paired_distance_metrics(raw_like_points, output_points)
    comparison = _path_comparison(output_metrics, raw_metrics)
    output_vs_tracking = _path_comparison(output_metrics, tracking_metrics)
    return {
        "raw_like_source": "processed.detections.lead_by_score",
        "tracking_source": "processed.confirmed_or_tentative.lead_by_track_rank",
        "output_source": output_source,
        "raw_like": raw_metrics,
        "tracking": tracking_metrics,
        "output": output_metrics,
        "trajectory_fidelity": fidelity,
        "output_vs_raw": comparison,
        "output_vs_tracking": output_vs_tracking,
        "policy": _policy(
            raw_metrics,
            output_metrics,
            fidelity,
            tracking=tracking_metrics,
            output_vs_tracking=output_vs_tracking,
        ),
        "straightness_diagnostic": _straightness_note(raw_metrics, output_metrics),
    }
