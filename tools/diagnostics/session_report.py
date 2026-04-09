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


def _safe_rate(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    try:
        denominator_value = float(denominator)
        numerator_value = float(numerator)
    except (TypeError, ValueError):
        return None
    if denominator_value <= 0:
        return None
    return numerator_value / denominator_value


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
    excluded_slowest_names = {"compute_total_ms", "pipeline_total_ms"}
    for stage_name, stats in timing_summary.items():
        if stage_name in excluded_slowest_names:
            continue
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


def _resolve_cfg_path(session_dir: Path, session_meta: dict, runtime_config: dict):
    project_root = session_meta.get("project_root")
    inferred_project_root = None
    try:
        inferred_project_root = session_dir.parents[2]
    except IndexError:
        inferred_project_root = None

    candidates = []
    for raw_value in (
        runtime_config.get("cfg"),
        (runtime_config.get("runtime_snapshot") or {}).get("config_path"),
    ):
        if not raw_value:
            continue
        raw_path = Path(str(raw_value))
        if raw_path.is_absolute():
            candidates.append(raw_path)
            continue
        if project_root:
            candidates.append(Path(project_root) / raw_path)
        if inferred_project_root is not None:
            candidates.append(inferred_project_root / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_frame_period_ms(config_path: Path | None):
    if config_path is None or not config_path.exists():
        return None
    with config_path.open(encoding="utf-8") as cfg_file:
        for raw_line in cfg_file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if parts and parts[0] == "frameCfg" and len(parts) >= 6:
                try:
                    return float(parts[5])
                except ValueError:
                    return None
    return None


def _lead_track_id(track_items):
    if not isinstance(track_items, list) or not track_items:
        return None

    def sort_key(item: dict):
        return (
            1 if item.get("is_primary") else 0,
            float(item.get("confidence") or 0.0),
            float(item.get("score") or 0.0),
            int(item.get("hits") or 0),
            int(item.get("age") or 0),
            -int(item.get("misses") or 0),
            float(item.get("rdi_peak") or 0.0),
        )

    best_item = max(track_items, key=sort_key)
    track_id = best_item.get("track_id")
    if track_id is None:
        return None
    try:
        return int(track_id)
    except (TypeError, ValueError):
        return None


def _lead_track_metrics(records: list[dict], list_key: str):
    previous_lead_id = None
    switches = 0
    unique_ids = set()
    lead_frame_count = 0

    for record in records:
        current_lead_id = _lead_track_id(record.get(list_key))
        if current_lead_id is None:
            continue
        lead_frame_count += 1
        unique_ids.add(current_lead_id)
        if previous_lead_id is not None and current_lead_id != previous_lead_id:
            switches += 1
        previous_lead_id = current_lead_id

    coverage_rate = _safe_rate(lead_frame_count, len(records))
    switch_rate = _safe_rate(switches, max(lead_frame_count - 1, 1)) if lead_frame_count > 1 else 0.0
    return {
        "frame_count_with_lead": lead_frame_count,
        "coverage_rate": _round_or_none(coverage_rate),
        "switch_count": switches,
        "switch_rate": _round_or_none(switch_rate),
        "unique_track_id_count": len(unique_ids),
    }


def _trajectory_point(item: dict, frame_id: int):
    if not isinstance(item, dict):
        return None
    x_m = item.get("x_m")
    y_m = item.get("y_m")
    if x_m is None or y_m is None:
        return None
    try:
        return {
            "frame_id": int(frame_id),
            "x_m": float(x_m),
            "y_m": float(y_m),
            "track_id": item.get("track_id"),
            "is_primary": bool(item.get("is_primary")),
            "confidence": float(item.get("confidence") or 0.0),
            "score": float(item.get("score") or 0.0),
            "hits": int(item.get("hits") or 0),
            "age": int(item.get("age") or 0),
            "misses": int(item.get("misses") or 0),
        }
    except (TypeError, ValueError):
        return None


def _track_item_rank(item: dict):
    return (
        1 if item.get("is_primary") else 0,
        float(item.get("confidence") or 0.0),
        float(item.get("score") or 0.0),
        int(item.get("hits") or 0),
        int(item.get("age") or 0),
        -int(item.get("misses") or 0),
    )


def _select_lead_point_from_record(record: dict, priority_keys: list[str]):
    frame_id = int(record.get("frame_id", record.get("frame_index", 0)) or 0)
    for key in priority_keys:
        items = record.get(key) or []
        if not items:
            continue
        lead = max(items, key=_track_item_rank)
        point = _trajectory_point(lead, frame_id)
        if point is not None:
            point["source_key"] = key
            return point
    return None


def _collect_lead_points(records: list[dict], priority_keys: list[str]):
    points = []
    source_hits = {key: 0 for key in priority_keys}
    for record in records:
        point = _select_lead_point_from_record(record, priority_keys)
        if point is None:
            continue
        source_key = point.get("source_key")
        if source_key in source_hits:
            source_hits[source_key] += 1
        points.append(point)
    points.sort(key=lambda item: item["frame_id"])
    source_key = None
    if source_hits and max(source_hits.values()) > 0:
        source_key = max(source_hits.items(), key=lambda item: item[1])[0]
    return points, source_key


def _distance_xy(left: dict, right: dict):
    return math.hypot(float(right["x_m"]) - float(left["x_m"]), float(right["y_m"]) - float(left["y_m"]))


def _point_to_segment_distance(point_xy, start_xy, end_xy):
    px, py = point_xy
    x1, y1 = start_xy
    x2, y2 = end_xy
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / ((dx * dx) + (dy * dy))
    t = min(1.0, max(0.0, t))
    proj_x = x1 + (t * dx)
    proj_y = y1 + (t * dy)
    return math.hypot(px - proj_x, py - proj_y)


def _build_path_geometry(points: list[dict], *, total_record_count: int, gap_break_frames: int = 2):
    if not points:
        return {
            "point_count": 0,
            "coverage_ratio": None,
            "frame_span": None,
            "segment_count": 0,
            "gap_count": 0,
            "max_gap_frames": 0,
            "path_length_m": None,
            "net_displacement_m": None,
            "path_efficiency_ratio": None,
            "x_span_m": None,
            "y_span_m": None,
            "step_length_p50_m": None,
            "step_length_p95_m": None,
            "normalized_step_p95_m": None,
            "jump_ratio": None,
            "local_residual_rms_m": None,
        }

    frame_ids = [int(point["frame_id"]) for point in points]
    coverage_ratio = _safe_rate(len(points), total_record_count)
    gap_count = 0
    max_gap_frames = 0
    segment_count = 1 if points else 0
    path_length_m = 0.0
    step_lengths = []
    normalized_steps = []

    for previous, current in zip(points, points[1:]):
        gap = int(current["frame_id"]) - int(previous["frame_id"])
        if gap > 1:
            gap_count += 1
            max_gap_frames = max(max_gap_frames, gap - 1)
            segment_count += 1
        if gap <= gap_break_frames:
            distance = _distance_xy(previous, current)
            path_length_m += distance
            step_lengths.append(distance)
            normalized_steps.append(distance / max(gap, 1))

    median_step = _quantile(normalized_steps, 0.50)
    jump_threshold = None
    if median_step is not None:
        jump_threshold = max(0.25, float(median_step) * 2.5)
    jump_ratio = None
    if normalized_steps:
        jump_events = sum(1 for step in normalized_steps if jump_threshold is not None and step > jump_threshold)
        jump_ratio = _safe_rate(jump_events, len(normalized_steps))

    local_residuals = []
    for previous, current, following in zip(points, points[1:], points[2:]):
        gap_left = int(current["frame_id"]) - int(previous["frame_id"])
        gap_right = int(following["frame_id"]) - int(current["frame_id"])
        if gap_left > gap_break_frames or gap_right > gap_break_frames:
            continue
        residual = _point_to_segment_distance(
            (float(current["x_m"]), float(current["y_m"])),
            (float(previous["x_m"]), float(previous["y_m"])),
            (float(following["x_m"]), float(following["y_m"])),
        )
        local_residuals.append(residual)

    local_residual_rms = None
    if local_residuals:
        local_residual_rms = math.sqrt(sum(value * value for value in local_residuals) / len(local_residuals))

    first_point = points[0]
    last_point = points[-1]
    net_displacement_m = _distance_xy(first_point, last_point) if len(points) >= 2 else 0.0
    path_efficiency_ratio = _safe_rate(net_displacement_m, path_length_m)

    xs = [float(point["x_m"]) for point in points]
    ys = [float(point["y_m"]) for point in points]

    return {
        "point_count": len(points),
        "coverage_ratio": _round_or_none(coverage_ratio),
        "frame_span": int(frame_ids[-1] - frame_ids[0] + 1) if len(frame_ids) >= 2 else 1,
        "segment_count": int(segment_count),
        "gap_count": int(gap_count),
        "max_gap_frames": int(max_gap_frames),
        "path_length_m": _round_or_none(path_length_m, digits=4),
        "net_displacement_m": _round_or_none(net_displacement_m, digits=4),
        "path_efficiency_ratio": _round_or_none(path_efficiency_ratio),
        "x_span_m": _round_or_none(max(xs) - min(xs), digits=4),
        "y_span_m": _round_or_none(max(ys) - min(ys), digits=4),
        "step_length_p50_m": _round_or_none(_quantile(step_lengths, 0.50), digits=4),
        "step_length_p95_m": _round_or_none(_quantile(step_lengths, 0.95), digits=4),
        "normalized_step_p95_m": _round_or_none(_quantile(normalized_steps, 0.95), digits=4),
        "jump_ratio": _round_or_none(jump_ratio),
        "local_residual_rms_m": _round_or_none(local_residual_rms, digits=4),
    }


def _unique_track_id_count(records: list[dict], list_key: str):
    unique_ids = set()
    for record in records:
        items = record.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            track_id = item.get("track_id")
            if track_id is None:
                continue
            try:
                unique_ids.add(int(track_id))
            except (TypeError, ValueError):
                continue
    return len(unique_ids)


def _p95_minus_p50(stats: dict):
    p95 = None if not isinstance(stats, dict) else stats.get("p95")
    p50 = None if not isinstance(stats, dict) else stats.get("p50")
    if p95 is None or p50 is None:
        return None
    return max(float(p95) - float(p50), 0.0)


def _score_from_anchors(value, anchors):
    if value is None:
        return None
    numeric_value = float(value)
    ordered = sorted((float(threshold), float(score)) for threshold, score in anchors)
    if numeric_value <= ordered[0][0]:
        return ordered[0][1]
    if numeric_value >= ordered[-1][0]:
        return ordered[-1][1]
    for (left_x, left_score), (right_x, right_score) in zip(ordered, ordered[1:]):
        if left_x <= numeric_value <= right_x:
            if right_x == left_x:
                return right_score
            ratio = (numeric_value - left_x) / (right_x - left_x)
            return left_score + ((right_score - left_score) * ratio)
    return ordered[-1][1]


def _score_tone(score_10):
    if score_10 is None:
        return "brand"
    if score_10 >= 8.5:
        return "good"
    if score_10 >= 7.0:
        return "brand"
    if score_10 >= 5.0:
        return "warn"
    return "danger"


def _score_band(score_10):
    if score_10 is None:
        return {"grade": "n/a", "label": "평가 불가", "tone": "brand"}
    if score_10 >= 9.0:
        return {"grade": "A", "label": "매우 양호", "tone": "good"}
    if score_10 >= 8.0:
        return {"grade": "B", "label": "양호", "tone": "good"}
    if score_10 >= 7.0:
        return {"grade": "C", "label": "보통", "tone": "brand"}
    if score_10 >= 6.0:
        return {"grade": "D", "label": "주의", "tone": "warn"}
    return {"grade": "F", "label": "미흡", "tone": "danger"}


def _weighted_average_score(items: list[tuple[float | None, float]]):
    weighted_total = 0.0
    total_weight = 0.0
    for score, weight in items:
        if score is None or weight <= 0:
            continue
        weighted_total += float(score) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0:
        return None
    return weighted_total / total_weight


def _kpi_entry(
    *,
    label: str,
    value,
    value_display: str,
    value_kind: str,
    score_10,
    target: str,
    calculation: str,
    meaning: str,
    industry_standard: str,
    interpretation: str,
):
    return {
        "label": label,
        "value": _round_or_none(value, digits=4) if isinstance(value, (int, float)) else value,
        "value_display": value_display,
        "value_kind": value_kind,
        "score_10": _round_or_none(score_10, digits=2),
        "score_100": _round_or_none(None if score_10 is None else score_10 * 10.0, digits=1),
        "tone": _score_tone(score_10),
        "target": target,
        "calculation": calculation,
        "meaning": meaning,
        "industry_standard": industry_standard,
        "interpretation": interpretation,
    }


def _performance_summary_text(score_10):
    if score_10 is None:
        return "성능 KPI를 계산할 데이터가 부족합니다."
    if score_10 >= 9.0:
        return "실시간 성능이 매우 안정적이며, 현업형 데모에 가까운 상태입니다."
    if score_10 >= 8.0:
        return "실시간 성능이 전반적으로 양호하며, 제한적 파일럿이나 데모에 적합한 수준입니다."
    if score_10 >= 7.0:
        return "기능 데모는 가능하지만 일부 KPI는 아직 보강이 필요합니다."
    if score_10 >= 6.0:
        return "동작은 하나 성능 여유와 continuity 품질이 충분하지 않습니다."
    return "성능 KPI 기준으로는 아직 실시간 체감과 안정성이 부족합니다."


def _build_performance_scoring(performance: dict):
    budget = performance.get("frame_budget") or {}
    throughput = performance.get("throughput") or {}
    compute = performance.get("compute") or {}
    jitter = performance.get("jitter") or {}
    continuity = performance.get("continuity") or {}
    geometry = performance.get("geometry") or {}
    geometry_reference = geometry.get("reference") or {}

    frame_period_ms = budget.get("configured_frame_period_ms")
    expected_fps = budget.get("expected_fps")
    processed_fps = throughput.get("processed_fps")
    render_fps = throughput.get("render_fps")
    processed_ratio = throughput.get("processed_vs_expected_ratio")
    render_ratio = throughput.get("render_vs_expected_ratio")
    compute_util_p95 = compute.get("compute_utilization_p95_ratio")
    compute_total_p95_ms = (compute.get("compute_total_ms") or {}).get("p95")
    render_latency_p95 = jitter.get("render_latency_p95_ms")
    render_jitter = jitter.get("render_latency_jitter_ms")
    candidate_ratio = continuity.get("candidate_to_confirmed_ratio")
    display_ratio = continuity.get("display_to_confirmed_ratio")
    lead_confirmed = continuity.get("lead_confirmed") or {}
    lead_switch_count = lead_confirmed.get("switch_count")
    lead_switch_rate = lead_confirmed.get("switch_rate")
    geometry_source = geometry.get("reference_source")
    path_cleanliness_score = geometry_reference.get("path_cleanliness_score_10")
    local_residual_rms = geometry_reference.get("local_residual_rms_m")
    max_gap_frames = geometry_reference.get("max_gap_frames")
    jump_ratio = geometry_reference.get("jump_ratio")

    processed_score = _score_from_anchors(
        processed_ratio,
        [(0.0, 0.0), (0.50, 3.0), (0.75, 6.0), (0.90, 8.0), (0.95, 9.0), (1.00, 10.0)],
    )
    render_score = _score_from_anchors(
        render_ratio,
        [(0.0, 0.0), (0.50, 3.0), (0.75, 6.0), (0.90, 8.0), (0.95, 9.0), (1.00, 10.0)],
    )
    compute_score = _score_from_anchors(
        compute_util_p95,
        [(0.35, 10.0), (0.50, 9.0), (0.70, 8.0), (0.85, 6.5), (1.00, 4.0), (1.20, 1.5), (1.50, 0.0)],
    )
    render_latency_score = _score_from_anchors(
        render_latency_p95,
        [(80.0, 10.0), (120.0, 9.0), (160.0, 8.0), (200.0, 6.5), (250.0, 4.5), (350.0, 2.0), (500.0, 0.0)],
    )
    render_jitter_score = _score_from_anchors(
        render_jitter,
        [(10.0, 10.0), (20.0, 9.0), (35.0, 7.5), (50.0, 6.0), (80.0, 3.0), (120.0, 0.0)],
    )
    candidate_score = _score_from_anchors(
        candidate_ratio,
        [(1.0, 10.0), (1.2, 9.0), (1.5, 7.5), (2.0, 5.5), (3.0, 2.5), (4.0, 0.0)],
    )
    display_score = _score_from_anchors(
        display_ratio,
        [(0.0, 0.0), (0.30, 3.5), (0.50, 6.0), (0.70, 8.0), (0.85, 9.0), (1.00, 10.0)],
    )
    lead_switch_score = _score_from_anchors(
        lead_switch_rate,
        [(0.0, 10.0), (0.03, 9.0), (0.07, 7.5), (0.12, 6.0), (0.20, 4.0), (0.35, 1.5), (0.50, 0.0)],
    )
    geometry_gap_score = _score_from_anchors(
        max_gap_frames,
        [(0.0, 10.0), (1.0, 9.5), (3.0, 8.0), (6.0, 6.0), (10.0, 4.0), (20.0, 1.5), (40.0, 0.0)],
    )
    geometry_residual_score = _score_from_anchors(
        local_residual_rms,
        [(0.02, 10.0), (0.04, 9.0), (0.07, 8.0), (0.12, 6.0), (0.20, 3.0), (0.35, 0.0)],
    )
    geometry_jump_score = _score_from_anchors(
        jump_ratio,
        [(0.0, 10.0), (0.03, 9.0), (0.07, 8.0), (0.15, 6.0), (0.30, 3.0), (0.50, 0.0)],
    )

    kpis = {
        "processed_vs_target": _kpi_entry(
            label="Processed FPS vs Target",
            value=processed_ratio,
            value_display=f"{_fmt_pct_ratio(processed_ratio)} ({_round_value_text(processed_fps, 'fps')} / 목표 {_round_value_text(expected_fps, 'fps')})",
            value_kind="ratio",
            score_10=processed_score,
            target="설정 목표 FPS의 95% 이상 권장, 100%면 설정과 동일한 처리량",
            calculation="processed_fps / expected_fps",
            meaning="처리 파이프라인이 설정된 목표 프레임률을 얼마나 따라갔는지 보여줍니다.",
            industry_standard="실시간 온라인 처리 기준으로 90~95% 이상이면 안정권, 80% 미만이면 프레임 누락 체감이 커질 수 있습니다.",
            interpretation=(
                f"실제 processed FPS는 {_round_value_text(processed_fps, 'fps')}이고 설정 목표는 {_round_value_text(expected_fps, 'fps')}입니다. "
                f"즉 목표 처리량의 {_fmt_pct_ratio(processed_ratio)}를 달성했다는 뜻입니다."
            ),
        ),
        "render_vs_target": _kpi_entry(
            label="Render FPS vs Target",
            value=render_ratio,
            value_display=f"{_fmt_pct_ratio(render_ratio)} ({_round_value_text(render_fps, 'fps')} / 목표 {_round_value_text(expected_fps, 'fps')})",
            value_kind="ratio",
            score_10=render_score,
            target="설정 목표 FPS의 90~95% 이상이면 화면 갱신이 자연스럽고, 100%면 목표와 동일한 표시량",
            calculation="render_fps / expected_fps",
            meaning="사용자가 실제로 본 화면 갱신률이 설정 목표를 얼마나 따라갔는지 보여줍니다.",
            industry_standard="인터랙티브 실시간 화면은 목표 대비 90% 이상 유지가 바람직합니다. 80% 아래면 눈에 띄는 끊김이 생기기 쉽습니다.",
            interpretation=(
                f"실제 render FPS는 {_round_value_text(render_fps, 'fps')}이고 설정 목표는 {_round_value_text(expected_fps, 'fps')}입니다. "
                f"즉 계획한 화면 갱신의 {_fmt_pct_ratio(render_ratio)}만 실제로 사용자가 봤다는 뜻입니다."
            ),
        ),
        "compute_utilization_p95": _kpi_entry(
            label="Compute Utilization P95",
            value=compute_util_p95,
            value_display=f"{_fmt_pct_ratio(compute_util_p95)} ({_round_value_text(compute_total_p95_ms, 'ms')} / 예산 {_round_value_text(frame_period_ms, 'ms')})",
            value_kind="ratio",
            score_10=compute_score,
            target="p95 기준 프레임 예산의 70% 이하 권장, 100%를 넘으면 계산만으로 예산 초과",
            calculation="compute_total_p95_ms / configured_frame_period_ms",
            meaning="느린 프레임 상위 5%에서 계산만으로 프레임 예산을 얼마나 쓰는지 보여줍니다.",
            industry_standard="실시간 파이프라인은 p95 기준 60~70% 이하가 안전합니다. 85%를 넘으면 환경 변화 시 급격히 나빠질 수 있습니다.",
            interpretation=(
                f"상위 95% 프레임에서 compute 구간은 {_round_value_text(compute_total_p95_ms, 'ms')}를 사용했습니다. "
                f"설정 프레임 예산 {_round_value_text(frame_period_ms, 'ms')} 대비 {_fmt_pct_ratio(compute_util_p95)}를 계산이 차지한다는 의미입니다."
            ),
        ),
        "render_latency_p95": _kpi_entry(
            label="Render Latency P95",
            value=render_latency_p95,
            value_display=_round_value_text(render_latency_p95, "ms"),
            value_kind="ms",
            score_10=render_latency_score,
            target="상위 95% 지연이 150~200ms 이내면 실시간 체감 가능, 250ms 이상이면 느리게 느껴질 수 있음",
            calculation="capture_to_render latency의 p95",
            meaning="입력 수집 시점부터 화면에 보이기 직전까지의 느린 프레임 상위 5% 지연입니다.",
            industry_standard="실시간 감시/추적 UI는 p95 기준 200ms 안팎이 보통 허용선이며, 250ms를 넘기면 체감 지연이 커집니다.",
            interpretation=(
                f"느린 프레임 상위 5%에서도 capture-to-render 지연이 {_round_value_text(render_latency_p95, 'ms')} 이내라는 뜻입니다. "
                f"평균보다 최악 구간 체감 품질을 보는 지표라 운영 판단에 중요합니다."
            ),
        ),
        "render_jitter": _kpi_entry(
            label="Render Jitter",
            value=render_jitter,
            value_display=_round_value_text(render_jitter, "ms"),
            value_kind="ms",
            score_10=render_jitter_score,
            target="p95 - p50 기준 20ms 이하가 이상적, 40ms 이상이면 세션 중 흔들림 체감 가능",
            calculation="render_latency_p95_ms - render_latency_p50_ms",
            meaning="같은 세션 안에서 render latency가 얼마나 흔들리는지 보여주는 변동폭입니다.",
            industry_standard="낮은 평균 지연보다 낮은 jitter가 더 중요할 때가 많습니다. 20ms 이하면 안정적, 40ms 이상이면 버벅임을 느끼기 쉽습니다.",
            interpretation=(
                f"현재 render latency 변동폭은 {_round_value_text(render_jitter, 'ms')}입니다. "
                "값이 클수록 어떤 프레임은 빠르고 어떤 프레임은 갑자기 늦어지는 세션이라는 뜻입니다."
            ),
        ),
        "candidate_to_confirmed": _kpi_entry(
            label="Candidate / Confirmed",
            value=candidate_ratio,
            value_display=_round_value_text(candidate_ratio),
            value_kind="ratio",
            score_10=candidate_score,
            target="1.3 이하 권장, 1.0에 가까울수록 한 사람을 한 detection/track으로 정리하는 데 유리",
            calculation="candidate_mean / confirmed_track_mean",
            meaning="confirmed track 1개를 만들기 위해 detection candidate가 몇 개나 나오고 있는지 보여줍니다.",
            industry_standard="단일 인원 테스트에서는 1.0~1.3 수준이 바람직합니다. 2.0 이상이면 한 사람을 여러 후보로 분해할 가능성이 큽니다.",
            interpretation=(
                f"현재 confirmed track 1개를 만들기 위해 평균적으로 {_round_value_text(candidate_ratio)}개의 candidate가 생깁니다. "
                "값이 높을수록 한 사람을 여러 detection으로 보고 있을 가능성이 큽니다."
            ),
        ),
        "display_to_confirmed": _kpi_entry(
            label="Display / Confirmed",
            value=display_ratio,
            value_display=_round_value_text(display_ratio),
            value_kind="ratio",
            score_10=display_score,
            target="0.8 이상 권장, 1.0에 가까울수록 내부 confirmed track이 화면까지 잘 전달됨",
            calculation="display_track_mean / confirmed_track_mean",
            meaning="내부 confirmed track이 실제 화면 표시까지 얼마나 살아남는지 보여줍니다.",
            industry_standard="실시간 데모 기준으로 내부 추적과 화면 표시가 크게 다르면 디버깅이 어렵습니다. 0.7~0.8 이상이 바람직합니다.",
            interpretation=(
                f"현재 display/confirmed 비율은 {_round_value_text(display_ratio)}입니다. "
                "내부에서는 잡았지만 화면에는 덜 보이는 비율이 크면 표시 정책이나 continuity 품질을 다시 봐야 합니다."
            ),
        ),
        "lead_confirmed_switch": _kpi_entry(
            label="Lead Confirmed Switch",
            value=lead_switch_count,
            value_display=f"{_round_value_text(lead_switch_count)}회 (rate {_fmt_pct_ratio(lead_switch_rate)})",
            value_kind="count",
            score_10=lead_switch_score,
            target="lead switch rate 5% 이하 권장. 단일 인원 세션에서는 대표 ID가 자주 바뀌지 않는 것이 이상적",
            calculation="lead switch count / (lead frame count - 1)",
            meaning="대표 confirmed ID가 세션 중 얼마나 자주 다른 ID로 바뀌었는지 보여줍니다.",
            industry_standard="단일 인원/단순 경로 테스트에서는 switch rate가 매우 낮아야 합니다. 10%를 넘기면 continuity 문제가 큰 편입니다.",
            interpretation=(
                f"대표 confirmed ID는 {_round_value_text(lead_switch_count)}회 바뀌었고, lead가 존재한 프레임 기준 switch rate는 {_fmt_pct_ratio(lead_switch_rate)}입니다. "
                "값이 높을수록 같은 사람을 하나의 ID로 오래 유지하지 못했다는 뜻입니다."
            ),
        ),
        "path_cleanliness": _kpi_entry(
            label="Path Cleanliness",
            value=path_cleanliness_score,
            value_display=f"{_round_value_text(path_cleanliness_score)}/10 ({geometry_source or 'n/a'})",
            value_kind="score",
            score_10=path_cleanliness_score,
            target="끊김, 지그재그 residual, 점프 비율이 모두 낮아 8점 이상이면 경로 품질이 양호한 편",
            calculation="weighted average(max_gap_frames, local_residual_rms_m, jump_ratio)",
            meaning="눈으로 봤을 때 경로가 얼마나 끊기지 않고, 과한 지그재그/점프 없이 이어지는지를 요약한 참고 점수입니다.",
            industry_standard="실제 이동 경로를 사람이 이해해야 하는 추적 UI에서는 continuity와 별도로 path cleanliness를 같이 봐야 합니다. 8점 이상이면 비교적 안정, 6점 이하면 눈으로 보기 거친 경우가 많습니다.",
            interpretation=(
                f"이번 세션의 기준 경로는 {geometry_source or 'n/a'}이고, 경로 청결도는 {_round_value_text(path_cleanliness_score)}/10 입니다. "
                "같은 ID를 유지하더라도 이 값이 낮으면 사람이 보기엔 경로가 말리거나 지저분하게 느껴질 수 있습니다."
            ),
        ),
        "path_max_gap_frames": _kpi_entry(
            label="Path Max Gap Frames",
            value=max_gap_frames,
            value_display=_round_value_text(max_gap_frames),
            value_kind="count",
            score_10=geometry_gap_score,
            target="0~2 frame 수준이면 양호, 5 frame 이상이면 눈으로도 끊김이 체감되기 쉬움",
            calculation="max(frame_id gap - 1) on lead path",
            meaning="대표 경로에서 가장 길게 비어 있는 프레임 수입니다.",
            industry_standard="실시간 경로 표시에서는 가장 긴 공백이 짧을수록 좋습니다. 단일 인원 경로에서 5프레임 이상 비면 사람이 보기에도 선이 끊겨 보이기 쉽습니다.",
            interpretation=(
                f"대표 경로의 최장 공백은 {_round_value_text(max_gap_frames)}프레임입니다. "
                "값이 클수록 코너나 약한 구간에서 경로가 뚝뚝 끊겨 보일 가능성이 큽니다."
            ),
        ),
        "path_local_residual_rms": _kpi_entry(
            label="Path Local Residual RMS",
            value=local_residual_rms,
            value_display=_round_value_text(local_residual_rms, "m"),
            value_kind="m",
            score_10=geometry_residual_score,
            target="0.05~0.10m 이하가 양호, 값이 커질수록 local zigzag가 커짐",
            calculation="RMS distance of lead points to local segment(prev->next)",
            meaning="대표 경로의 각 점이 바로 앞뒤 점이 만드는 local segment에서 얼마나 벗어나는지 보는 지터 지표입니다.",
            industry_standard="사람 이동 궤적을 눈으로 이해해야 하는 응용에서는 수 cm~10 cm대 residual이 보통 양호합니다. 0.15m를 넘기면 선이 말리거나 뱀처럼 흔들려 보이기 쉽습니다.",
            interpretation=(
                f"현재 local residual RMS는 {_round_value_text(local_residual_rms, 'm')}입니다. "
                "값이 높을수록 전체 ID는 유지돼도 경로가 지그재그로 흔들려 보입니다."
            ),
        ),
        "path_jump_ratio": _kpi_entry(
            label="Path Jump Ratio",
            value=jump_ratio,
            value_display=_fmt_pct_ratio(jump_ratio),
            value_kind="ratio",
            score_10=geometry_jump_score,
            target="3~7% 이하가 양호, 커질수록 순간적인 좌표 점프가 많음",
            calculation="ratio of normalized steps larger than adaptive jump threshold",
            meaning="대표 경로에서 과하게 큰 step이 전체 step 중 얼마나 자주 나타나는지 보는 지표입니다.",
            industry_standard="직선/왕복/원형 등 대부분의 단일 인원 경로에서는 큰 점프가 드물어야 합니다. 10%를 넘기면 눈에 띄는 튐으로 느껴질 가능성이 높습니다.",
            interpretation=(
                f"현재 jump ratio는 {_fmt_pct_ratio(jump_ratio)}입니다. "
                "값이 높을수록 일부 프레임에서 대표점이 갑자기 다른 위치로 뛰는 현상이 잦다는 의미입니다."
            ),
        ),
    }

    categories = {
        "throughput": _weighted_average_score(
            [
                (processed_score, 1.0),
                (render_score, 1.0),
                (render_latency_score, 1.0),
            ]
        ),
        "efficiency": _weighted_average_score([(compute_score, 1.0)]),
        "stability": _weighted_average_score([(render_jitter_score, 1.0)]),
        "continuity": _weighted_average_score(
            [
                (candidate_score, 1.0),
                (display_score, 1.0),
                (lead_switch_score, 1.2),
            ]
        ),
        "geometry": _weighted_average_score(
            [
                (geometry_gap_score, 1.1),
                (geometry_residual_score, 1.0),
                (geometry_jump_score, 0.9),
            ]
        ),
    }

    category_labels = {
        "throughput": "처리량/목표 달성",
        "efficiency": "계산 여유",
        "stability": "지연 안정성",
        "continuity": "추적 연속성",
        "geometry": "경로 기하 품질",
    }
    category_scores = {}
    for key, score in categories.items():
        band = _score_band(score)
        category_scores[key] = {
            "label": category_labels[key],
            "score_10": _round_or_none(score, digits=2),
            "score_100": _round_or_none(None if score is None else score * 10.0, digits=1),
            "grade": band["grade"],
            "tone": band["tone"],
        }

    overall_score_10 = _weighted_average_score(
        [
            (processed_score, 1.0),
            (render_score, 1.0),
            (compute_score, 1.0),
            (render_latency_score, 1.0),
            (render_jitter_score, 1.0),
            (candidate_score, 1.25),
            (display_score, 1.25),
            (lead_switch_score, 1.5),
            (geometry_gap_score, 1.25),
            (geometry_residual_score, 1.0),
            (geometry_jump_score, 1.0),
        ]
    )
    band = _score_band(overall_score_10)
    return {
        "overall_score_10": _round_or_none(overall_score_10, digits=2),
        "overall_score_100": _round_or_none(None if overall_score_10 is None else overall_score_10 * 10.0, digits=1),
        "grade": band["grade"],
        "label": band["label"],
        "tone": band["tone"],
        "summary": _performance_summary_text(overall_score_10),
        "categories": category_scores,
        "kpis": kpis,
    }


def _fmt_pct_ratio(value):
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _round_value_text(value, unit: str = ""):
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{float(value):.2f}".rstrip("0").rstrip(".") + suffix


def _build_performance_summary(
    session_dir: Path,
    session_meta: dict,
    runtime_config: dict,
    event_summary: dict,
    processed_records: list[dict],
    render_records: list[dict],
    processed_summary: dict,
    render_summary: dict,
    preferred_stage_timings: dict,
):
    cfg_path = _resolve_cfg_path(session_dir, session_meta, runtime_config)
    frame_period_ms = _read_frame_period_ms(cfg_path)
    expected_fps = (1000.0 / frame_period_ms) if frame_period_ms and frame_period_ms > 0 else None
    session_duration_s = event_summary.get("session_duration_s")

    processed_fps = _safe_rate(processed_summary.get("frame_count"), session_duration_s)
    render_fps = _safe_rate(render_summary.get("frame_count"), session_duration_s)
    render_to_processed_ratio = _safe_rate(render_fps, processed_fps)
    processed_vs_expected_ratio = _safe_rate(processed_fps, expected_fps)
    render_vs_expected_ratio = _safe_rate(render_fps, expected_fps)

    preferred_timing_map = (preferred_stage_timings or {}).get("timings") or {}
    compute_stats = preferred_timing_map.get("compute_total_ms") or {}
    pipeline_stats = preferred_timing_map.get("pipeline_total_ms") or {}
    log_write_stats = preferred_timing_map.get("log_write_ms") or {}
    slowest_stage = (preferred_stage_timings or {}).get("slowest_stage") or {}

    compute_mean = compute_stats.get("mean")
    compute_p50 = compute_stats.get("p50")
    compute_p95 = compute_stats.get("p95")
    pipeline_mean = pipeline_stats.get("mean")
    pipeline_p95 = pipeline_stats.get("p95")
    slowest_stage_p95 = slowest_stage.get("p95_ms")

    compute_utilization_mean = _safe_rate(compute_mean, frame_period_ms)
    compute_utilization_p95 = _safe_rate(compute_p95, frame_period_ms)
    pipeline_utilization_mean = _safe_rate(pipeline_mean, frame_period_ms)
    pipeline_utilization_p95 = _safe_rate(pipeline_p95, frame_period_ms)
    slowest_stage_share = _safe_rate(slowest_stage_p95, compute_p95)

    non_compute_mean = None
    non_compute_p95 = None
    processed_latency_stats = processed_summary.get("capture_to_process_ms") or {}
    if processed_latency_stats.get("mean") is not None and compute_mean is not None:
        non_compute_mean = max(float(processed_latency_stats["mean"]) - float(compute_mean), 0.0)
    if processed_latency_stats.get("p95") is not None and compute_p95 is not None:
        non_compute_p95 = max(float(processed_latency_stats["p95"]) - float(compute_p95), 0.0)

    render_latency_stats = render_summary.get("capture_to_render_ms") or {}
    process_to_render_stats = render_summary.get("process_to_render_ms") or {}

    confirmed_lead_metrics = _lead_track_metrics(processed_records, "confirmed_tracks")
    display_lead_metrics = _lead_track_metrics(render_records, "display_tracks")

    candidate_mean = (processed_summary.get("candidate_count") or {}).get("mean")
    confirmed_mean = (processed_summary.get("confirmed_track_count") or {}).get("mean")
    display_mean = (render_summary.get("display_track_count") or {}).get("mean")

    candidate_to_confirmed_ratio = _safe_rate(candidate_mean, confirmed_mean)
    display_to_confirmed_ratio = _safe_rate(display_mean, confirmed_mean)

    render_geometry_points, render_geometry_source = _collect_lead_points(
        render_records,
        ["display_tracks", "tentative_display_tracks", "tentative_tracks"],
    )
    processed_geometry_points, processed_geometry_source = _collect_lead_points(
        processed_records,
        ["confirmed_tracks", "tentative_tracks"],
    )
    render_geometry = _build_path_geometry(
        render_geometry_points,
        total_record_count=len(render_records),
        gap_break_frames=2,
    )
    processed_geometry = _build_path_geometry(
        processed_geometry_points,
        total_record_count=len(processed_records),
        gap_break_frames=2,
    )

    def attach_geometry_scores(geometry_stats: dict):
        gap_score = _score_from_anchors(
            geometry_stats.get("max_gap_frames"),
            [(0.0, 10.0), (1.0, 9.5), (3.0, 8.0), (6.0, 6.0), (10.0, 4.0), (20.0, 1.5), (40.0, 0.0)],
        )
        residual_score = _score_from_anchors(
            geometry_stats.get("local_residual_rms_m"),
            [(0.02, 10.0), (0.04, 9.0), (0.07, 8.0), (0.12, 6.0), (0.20, 3.0), (0.35, 0.0)],
        )
        jump_score = _score_from_anchors(
            geometry_stats.get("jump_ratio"),
            [(0.0, 10.0), (0.03, 9.0), (0.07, 8.0), (0.15, 6.0), (0.30, 3.0), (0.50, 0.0)],
        )
        cleanliness = _weighted_average_score(
            [
                (gap_score, 1.1),
                (residual_score, 1.0),
                (jump_score, 0.9),
            ]
        )
        scored = dict(geometry_stats)
        scored["path_cleanliness_score_10"] = _round_or_none(cleanliness, digits=2)
        scored["path_cleanliness_score_100"] = _round_or_none(
            None if cleanliness is None else cleanliness * 10.0,
            digits=1,
        )
        return scored

    render_geometry = attach_geometry_scores(render_geometry)
    processed_geometry = attach_geometry_scores(processed_geometry)

    geometry_reference_source = "processed_lead"
    geometry_reference = processed_geometry
    if (render_geometry.get("point_count") or 0) >= 3:
        geometry_reference_source = "render_lead"
        geometry_reference = render_geometry

    geometry_gap_score = _score_from_anchors(
        geometry_reference.get("max_gap_frames"),
        [(0.0, 10.0), (1.0, 9.5), (3.0, 8.0), (6.0, 6.0), (10.0, 4.0), (20.0, 1.5), (40.0, 0.0)],
    )
    geometry_residual_score = _score_from_anchors(
        geometry_reference.get("local_residual_rms_m"),
        [(0.02, 10.0), (0.04, 9.0), (0.07, 8.0), (0.12, 6.0), (0.20, 3.0), (0.35, 0.0)],
    )
    geometry_jump_score = _score_from_anchors(
        geometry_reference.get("jump_ratio"),
        [(0.0, 10.0), (0.03, 9.0), (0.07, 8.0), (0.15, 6.0), (0.30, 3.0), (0.50, 0.0)],
    )
    path_cleanliness_score_10 = _weighted_average_score(
        [
            (geometry_gap_score, 1.1),
            (geometry_residual_score, 1.0),
            (geometry_jump_score, 0.9),
        ]
    )

    geometry_reference_payload = dict(geometry_reference)

    performance = {
        "frame_budget": {
            "cfg_path": None if cfg_path is None else str(cfg_path),
            "configured_frame_period_ms": _round_or_none(frame_period_ms, digits=3),
            "expected_fps": _round_or_none(expected_fps, digits=3),
        },
        "throughput": {
            "session_duration_s": _round_or_none(session_duration_s, digits=3),
            "processed_fps": _round_or_none(processed_fps, digits=3),
            "render_fps": _round_or_none(render_fps, digits=3),
            "render_to_processed_ratio": _round_or_none(render_to_processed_ratio, digits=3),
            "processed_vs_expected_ratio": _round_or_none(processed_vs_expected_ratio, digits=3),
            "render_vs_expected_ratio": _round_or_none(render_vs_expected_ratio, digits=3),
        },
        "compute": {
            "source": preferred_stage_timings.get("source"),
            "frames_with_timings": preferred_stage_timings.get("frame_count_with_timings", 0),
            "compute_total_ms": compute_stats,
            "pipeline_total_ms": pipeline_stats,
            "log_write_ms": log_write_stats,
            "compute_utilization_mean_ratio": _round_or_none(compute_utilization_mean, digits=3),
            "compute_utilization_p95_ratio": _round_or_none(compute_utilization_p95, digits=3),
            "pipeline_utilization_mean_ratio": _round_or_none(pipeline_utilization_mean, digits=3),
            "pipeline_utilization_p95_ratio": _round_or_none(pipeline_utilization_p95, digits=3),
            "non_compute_capture_to_process_mean_ms": _round_or_none(non_compute_mean, digits=3),
            "non_compute_capture_to_process_p95_ms": _round_or_none(non_compute_p95, digits=3),
            "render_overhead_mean_ms": process_to_render_stats.get("mean"),
            "render_overhead_p95_ms": process_to_render_stats.get("p95"),
            "slowest_stage_name": slowest_stage.get("name"),
            "slowest_stage_p95_ms": slowest_stage_p95,
            "slowest_stage_share_of_compute_p95_ratio": _round_or_none(slowest_stage_share, digits=3),
        },
        "jitter": {
            "processed_latency_p50_ms": processed_latency_stats.get("p50"),
            "processed_latency_p95_ms": processed_latency_stats.get("p95"),
            "processed_latency_jitter_ms": _round_or_none(_p95_minus_p50(processed_latency_stats), digits=3),
            "render_latency_p50_ms": render_latency_stats.get("p50"),
            "render_latency_p95_ms": render_latency_stats.get("p95"),
            "render_latency_jitter_ms": _round_or_none(_p95_minus_p50(render_latency_stats), digits=3),
            "compute_total_p50_ms": compute_p50,
            "compute_total_p95_ms": compute_p95,
            "compute_total_jitter_ms": _round_or_none(_p95_minus_p50(compute_stats), digits=3),
        },
        "continuity": {
            "candidate_to_confirmed_ratio": _round_or_none(candidate_to_confirmed_ratio, digits=3),
            "display_to_confirmed_ratio": _round_or_none(display_to_confirmed_ratio, digits=3),
            "lead_confirmed": confirmed_lead_metrics,
            "lead_display": display_lead_metrics,
            "unique_confirmed_track_ids": _unique_track_id_count(processed_records, "confirmed_tracks"),
            "unique_display_track_ids": _unique_track_id_count(render_records, "display_tracks"),
        },
        "geometry": {
            "reference_source": geometry_reference_source,
            "reference": geometry_reference_payload,
            "render_lead": {
                "source_key": render_geometry_source,
                **render_geometry,
            },
            "processed_lead": {
                "source_key": processed_geometry_source,
                **processed_geometry,
            },
        },
        "highlights": [],
    }
    performance["scoring"] = _build_performance_scoring(performance)

    highlights = performance["highlights"]
    if compute_utilization_p95 is not None:
        if compute_utilization_p95 <= 0.60:
            highlights.append("순수 compute p95는 프레임 예산의 60% 이하로, 계산 자체는 아직 여유가 있습니다.")
        elif compute_utilization_p95 >= 0.95:
            highlights.append("compute p95가 프레임 예산에 거의 닿아 계산 병목 위험이 큽니다.")
    if render_vs_expected_ratio is not None and render_vs_expected_ratio < 0.80:
        highlights.append("렌더 FPS가 설정 frame rate를 충분히 따라가지 못해 화면 갱신 효율이 낮습니다.")
    if candidate_to_confirmed_ratio is not None and candidate_to_confirmed_ratio >= 1.5:
        highlights.append("candidate 대비 confirmed 비율이 낮아, detection 분해 또는 track 정리 품질을 우선 의심해야 합니다.")
    if confirmed_lead_metrics.get("switch_count", 0) >= 10:
        highlights.append("lead confirmed ID switch가 많아 continuity 품질이 아직 약합니다.")
    if _p95_minus_p50(render_latency_stats) not in (None, 0) and _p95_minus_p50(render_latency_stats) >= 30:
        highlights.append("render latency 변동폭이 커서 체감 품질이 세션 중 흔들릴 수 있습니다.")
    if (geometry_reference.get("max_gap_frames") or 0) >= 5:
        highlights.append("lead path가 길게 비는 구간이 있어, 화면에서 경로가 끊겨 보일 가능성이 큽니다.")
    if (geometry_reference.get("local_residual_rms_m") or 0.0) >= 0.12:
        highlights.append("lead path의 local residual이 커, 직선/대각선 테스트에서도 지그재그가 눈에 띌 수 있습니다.")
    if (geometry_reference.get("jump_ratio") or 0.0) >= 0.10:
        highlights.append("일부 프레임에서 대표점이 갑자기 점프하는 비율이 높아, 코너나 방향 전환에서 경로가 말릴 수 있습니다.")
    if not highlights:
        highlights.append("핵심 KPI는 대체로 안정적이며, 다음 비교 세션과의 추세 확인이 중요합니다.")

    return performance


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
    render_held_display_counts = [
        int(record.get("display_held_track_count", 0))
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
        "schema_version": 4,
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
            "display_held_track_count": _summarize_numeric(render_held_display_counts, digits=3),
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
    summary["performance"] = _build_performance_summary(
        session_dir,
        session_meta,
        runtime_config,
        summary["event"],
        processed_records,
        render_records,
        summary["processed"],
        summary["render"],
        preferred_stage_timings,
    )
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
