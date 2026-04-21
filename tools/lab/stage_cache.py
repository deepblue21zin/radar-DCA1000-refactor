from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from tools.lab import registry
from tools.runtime_core.detection import DetectionRegion
from tools.runtime_core.radar_runtime import parse_runtime_config, radial_bin_limit
from tools.runtime_core.real_time_process import (
    _serialize_detection,
    _serialize_track,
    iter_raw_capture_frame_packets,
    load_raw_capture,
    process_frame_packet,
)
from tools.runtime_core.tracking import MultiTargetTracker


STAGE_CACHE_SCHEMA_VERSION = 2
STAGE_FEATURE_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _nested_get(payload: dict | None, *keys: str, default=None):
    current = payload or {}
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _as_path(project_root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _resolve_capture_dir(project_root: Path, run_detail: dict) -> Path:
    capture_candidates = []
    if run_detail.get("capture_id"):
        capture_candidates.append(project_root / "logs" / "raw" / str(run_detail["capture_id"]))

    session_meta = run_detail.get("session_meta") or {}
    runtime_summary = run_detail.get("runtime_config") or {}
    summary = run_detail.get("summary") or {}

    capture_candidates.extend(
        candidate
        for candidate in [
            _as_path(project_root, session_meta.get("raw_capture_dir")),
            _as_path(project_root, session_meta.get("source_capture")),
            _as_path(project_root, runtime_summary.get("raw_capture_dir")),
            _as_path(project_root, runtime_summary.get("log_source_capture")),
            _as_path(project_root, _nested_get(summary, "session_meta", "raw_capture_dir")),
            _as_path(project_root, _nested_get(summary, "runtime_config", "raw_capture_dir")),
        ]
        if candidate is not None
    )

    for candidate in capture_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "This run does not have a resolvable raw capture directory. "
        "Check capture linking or record a session with raw capture enabled."
    )


def _merge_runtime_summary(run_detail: dict, capture_manifest: dict) -> dict:
    merged: dict = {}
    for payload in [
        capture_manifest.get("runtime_summary"),
        _nested_get(run_detail, "summary", "runtime_config"),
        run_detail.get("runtime_config"),
    ]:
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def _resolve_cfg_path(project_root: Path, runtime_summary: dict) -> Path:
    cfg_path = (
        runtime_summary.get("cfg")
        or _nested_get(runtime_summary, "runtime_snapshot", "config_path")
        or _nested_get(runtime_summary, "raw_capture", "config_path")
    )
    resolved = _as_path(project_root, cfg_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Could not resolve cfg path from runtime summary: {cfg_path}")
    return resolved


def _estimate_angle_resolution_rad(runtime_config) -> float:
    angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=float)
    if angle_axis.size < 2:
        return 0.0
    diffs = np.diff(angle_axis)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(np.abs(diffs)))


def _build_runtime_components(project_root: Path, runtime_summary: dict):
    cfg_path = _resolve_cfg_path(project_root, runtime_summary)
    remove_static = bool(
        runtime_summary.get(
            "remove_static",
            _nested_get(runtime_summary, "tuning_snapshot", "processing", "remove_static", default=True),
        )
    )
    doppler_guard_bins = int(
        _nested_get(runtime_summary, "tuning_snapshot", "processing", "doppler_guard_bins", default=1)
    )
    lateral_axis_sign = runtime_summary.get("lateral_axis_sign")
    if lateral_axis_sign is None:
        lateral_axis_sign = -1.0 if runtime_summary.get("invert_lateral_axis") else 1.0

    runtime_config = parse_runtime_config(
        cfg_path,
        remove_static=remove_static,
        doppler_guard_bins=doppler_guard_bins,
        lateral_axis_sign=float(lateral_axis_sign),
    )

    roi_lateral_m = float(
        runtime_summary.get(
            "roi_lateral_m",
            _nested_get(runtime_summary, "tuning_snapshot", "roi", "lateral_m", default=1.5),
        )
    )
    roi_forward_m = float(
        runtime_summary.get(
            "roi_forward_m",
            _nested_get(runtime_summary, "tuning_snapshot", "roi", "forward_m", default=4.0),
        )
    )
    roi_min_forward_m = float(
        runtime_summary.get(
            "roi_min_forward_m",
            _nested_get(runtime_summary, "tuning_snapshot", "roi", "min_forward_m", default=0.0),
        )
    )
    detection_region = DetectionRegion(
        lateral_limit_m=roi_lateral_m,
        forward_limit_m=roi_forward_m,
        min_forward_m=roi_min_forward_m,
        max_targets=int(
            runtime_summary.get(
                "max_targets",
                _nested_get(runtime_summary, "tuning_snapshot", "detection", "max_targets", default=6),
            )
        ),
        allow_strongest_fallback=bool(
            runtime_summary.get(
                "allow_strongest_fallback",
                _nested_get(
                    runtime_summary,
                    "tuning_snapshot",
                    "detection",
                    "allow_strongest_fallback",
                    default=False,
                ),
            )
        ),
        adaptive_eps_bands=runtime_summary.get(
            "dbscan_adaptive_eps_bands",
            _nested_get(runtime_summary, "tuning_snapshot", "detection", "dbscan_adaptive_eps_bands"),
        ),
        cluster_min_samples=int(
            runtime_summary.get(
                "cluster_min_samples",
                _nested_get(runtime_summary, "tuning_snapshot", "detection", "cluster_min_samples", default=1),
            )
        ),
        cluster_velocity_weight=float(
            runtime_summary.get(
                "cluster_velocity_weight",
                _nested_get(
                    runtime_summary,
                    "tuning_snapshot",
                    "detection",
                    "cluster_velocity_weight",
                    default=0.0,
                ),
            )
        ),
    )

    detection_params = dict(
        runtime_summary.get(
            "detection_algorithm",
            _nested_get(runtime_summary, "tuning_snapshot", "detection", "algorithm", default={}),
        )
        or {}
    )
    if "cfar_training_cells" in detection_params:
        detection_params["cfar_training_cells"] = tuple(detection_params["cfar_training_cells"])
    if "cfar_guard_cells" in detection_params:
        detection_params["cfar_guard_cells"] = tuple(detection_params["cfar_guard_cells"])

    angle_resolution_deg = runtime_summary.get("track_angle_resolution_deg")
    if angle_resolution_deg is not None:
        angle_resolution_rad = math.radians(float(angle_resolution_deg))
    else:
        angle_resolution_rad = _estimate_angle_resolution_rad(runtime_config)

    tracker = MultiTargetTracker(
        process_var=float(runtime_summary.get("track_process_var", 1.0)),
        measurement_var=float(runtime_summary.get("track_measurement_var", 0.4)),
        range_measurement_scale=float(runtime_summary.get("track_range_measurement_scale", 0.0)),
        confidence_measurement_scale=float(runtime_summary.get("track_confidence_measurement_scale", 0.0)),
        angle_resolution_rad=angle_resolution_rad,
        association_gate=float(runtime_summary.get("track_association_gate", 5.99)),
        doppler_center_bin=int(runtime_config.doppler_fft_size // 2),
        doppler_zero_guard_bins=int(runtime_summary.get("track_doppler_zero_guard_bins", 2)),
        doppler_gate_bins=int(runtime_summary.get("track_doppler_gate_bins", 0)),
        doppler_cost_weight=float(runtime_summary.get("track_doppler_cost_weight", 0.0)),
        min_confirmed_hits=int(runtime_summary.get("track_confirm_hits", 2)),
        max_missed_frames=int(runtime_summary.get("track_max_misses", 8)),
        report_miss_tolerance=int(runtime_summary.get("track_report_miss_tolerance", 2)),
        lost_gate_factor=float(runtime_summary.get("track_lost_gate_factor", 1.2)),
        tentative_gate_factor=float(runtime_summary.get("track_tentative_gate_factor", 0.5)),
        birth_suppression_radius_m=float(runtime_summary.get("track_birth_suppression_radius_m", 0.0)),
        primary_track_birth_scale=float(runtime_summary.get("track_primary_track_birth_scale", 1.0)),
        birth_suppression_miss_tolerance=int(
            runtime_summary.get("track_birth_suppression_miss_tolerance", 0)
        ),
        primary_track_hold_frames=int(runtime_summary.get("track_primary_track_hold_frames", 0)),
        lateral_deadband_m=float(runtime_summary.get("track_lateral_deadband_m", 0.0)),
        lateral_deadband_range_scale=float(runtime_summary.get("track_lateral_deadband_range_scale", 0.0)),
        lateral_smoothing_alpha=float(runtime_summary.get("track_lateral_smoothing_alpha", 1.0)),
        lateral_velocity_damping=float(runtime_summary.get("track_lateral_velocity_damping", 1.0)),
        local_remeasurement_enabled=bool(
            runtime_summary.get("track_local_remeasurement_enabled", False)
        ),
        local_remeasurement_blend=float(runtime_summary.get("track_local_remeasurement_blend", 0.0)),
        local_remeasurement_max_shift_m=float(
            runtime_summary.get("track_local_remeasurement_max_shift_m", 0.0)
        ),
        local_remeasurement_track_bias=float(
            runtime_summary.get("track_local_remeasurement_track_bias", 0.0)
        ),
        local_remeasurement_patch_bands=runtime_summary.get("track_local_remeasurement_patch_bands"),
        measurement_soft_gate_enabled=bool(
            runtime_summary.get("track_measurement_soft_gate_enabled", True)
        ),
        measurement_soft_gate_floor=float(runtime_summary.get("track_measurement_soft_gate_floor", 0.35)),
        measurement_soft_gate_start_m=float(runtime_summary.get("track_measurement_soft_gate_start_m", 0.16)),
        measurement_soft_gate_full_m=float(runtime_summary.get("track_measurement_soft_gate_full_m", 0.52)),
        measurement_soft_gate_range_scale=float(
            runtime_summary.get("track_measurement_soft_gate_range_scale", 0.05)
        ),
        measurement_soft_gate_speed_scale=float(
            runtime_summary.get("track_measurement_soft_gate_speed_scale", 0.06)
        ),
    )

    min_range_bin = (
        radial_bin_limit(runtime_config, roi_min_forward_m) if float(roi_min_forward_m) > 0 else 0
    )
    max_range_bin = radial_bin_limit(
        runtime_config,
        math.sqrt((roi_lateral_m ** 2) + (roi_forward_m ** 2)),
    )
    invalid_policy = runtime_summary.get("invalid_policy") or {}
    block_track_birth_on_invalid = bool(runtime_summary.get("block_track_birth_on_invalid", True))
    return {
        "cfg_path": cfg_path,
        "runtime_config": runtime_config,
        "detection_region": detection_region,
        "detection_params": detection_params,
        "tracker": tracker,
        "min_range_bin": int(min_range_bin),
        "max_range_bin": int(max_range_bin),
        "invalid_policy": invalid_policy,
        "block_track_birth_on_invalid": block_track_birth_on_invalid,
        "roi_lateral_m": roi_lateral_m,
        "roi_forward_m": roi_forward_m,
        "roi_min_forward_m": roi_min_forward_m,
    }


def stage_cache_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "lab_data" / "stage_cache"


def stage_cache_dir(project_root: Path, session_id: str) -> Path:
    return stage_cache_root(project_root) / str(session_id)


def stage_cache_paths(project_root: Path, session_id: str) -> dict[str, Path]:
    cache_dir = stage_cache_dir(project_root, session_id)
    return {
        "cache_dir": cache_dir,
        "manifest_path": cache_dir / "manifest.json",
        "frames_path": cache_dir / "frames.jsonl",
        "features_path": cache_dir / "frame_features.jsonl",
        "feature_summary_path": cache_dir / "feature_summary.json",
        "trace_path": cache_dir / "frame_trace.jsonl",
        "trace_summary_path": cache_dir / "trace_summary.json",
        "artifacts_dir": cache_dir / "artifacts",
    }


def load_stage_cache_manifest(project_root: Path, session_id: str) -> dict | None:
    manifest_path = stage_cache_paths(project_root, session_id)["manifest_path"]
    if not manifest_path.exists():
        return None
    return _load_json(manifest_path)


def load_stage_cache_frames(project_root: Path, session_id: str) -> list[dict]:
    frames_path = stage_cache_paths(project_root, session_id)["frames_path"]
    return _load_jsonl(frames_path)


def load_stage_features(project_root: Path, session_id: str) -> list[dict]:
    features_path = stage_cache_paths(project_root, session_id)["features_path"]
    return _load_jsonl(features_path)


def load_stage_feature_summary(project_root: Path, session_id: str) -> dict | None:
    summary_path = stage_cache_paths(project_root, session_id)["feature_summary_path"]
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def load_stage_traces(project_root: Path, session_id: str) -> list[dict]:
    trace_path = stage_cache_paths(project_root, session_id)["trace_path"]
    return _load_jsonl(trace_path)


def load_stage_trace_summary(project_root: Path, session_id: str) -> dict | None:
    summary_path = stage_cache_paths(project_root, session_id)["trace_summary_path"]
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def load_stage_cache_frame(project_root: Path, session_id: str, ordinal: int) -> tuple[dict, dict[str, np.ndarray]]:
    cache_dir = stage_cache_dir(project_root, session_id)
    frames = load_stage_cache_frames(project_root, session_id)
    for record in frames:
        if int(record.get("ordinal", -1)) != int(ordinal):
            continue
        artifact_rel = record.get("artifact_file")
        if not artifact_rel:
            raise FileNotFoundError(f"Frame {ordinal} does not have a stored artifact file.")
        artifact_path = cache_dir / artifact_rel
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file missing: {artifact_path}")
        with np.load(artifact_path) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        return record, arrays
    raise IndexError(f"Frame ordinal {ordinal} is not present in the stage cache.")


def _clear_existing_cache(paths: dict[str, Path]) -> None:
    frames_path = paths["frames_path"]
    manifest_path = paths["manifest_path"]
    features_path = paths["features_path"]
    feature_summary_path = paths["feature_summary_path"]
    trace_path = paths["trace_path"]
    trace_summary_path = paths["trace_summary_path"]
    artifacts_dir = paths["artifacts_dir"]
    if frames_path.exists():
        try:
            frames_path.unlink()
        except PermissionError:
            frames_path.write_text("", encoding="utf-8")
    if features_path.exists():
        try:
            features_path.unlink()
        except PermissionError:
            features_path.write_text("", encoding="utf-8")
    if feature_summary_path.exists():
        try:
            feature_summary_path.unlink()
        except PermissionError:
            feature_summary_path.write_text("{}", encoding="utf-8")
    if trace_path.exists():
        try:
            trace_path.unlink()
        except PermissionError:
            trace_path.write_text("", encoding="utf-8")
    if trace_summary_path.exists():
        try:
            trace_summary_path.unlink()
        except PermissionError:
            trace_summary_path.write_text("{}", encoding="utf-8")
    if manifest_path.exists():
        try:
            manifest_path.unlink()
        except PermissionError:
            manifest_path.write_text("{}", encoding="utf-8")
    if artifacts_dir.exists():
        for artifact in artifacts_dir.glob("*.npz"):
            try:
                artifact.unlink()
            except PermissionError:
                continue
    artifacts_dir.mkdir(parents=True, exist_ok=True)


def _round_or_none(value, digits=4):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, digits)


def _array_quality_stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "max": None,
            "mean": None,
            "p95": None,
            "median": None,
            "peak_to_median": None,
            "active_ratio": None,
        }
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    flat = np.ravel(array)
    peak = float(np.max(flat))
    median = float(np.median(flat))
    active_threshold = float(np.percentile(flat, 90)) if flat.size else 0.0
    return {
        "max": _round_or_none(peak),
        "mean": _round_or_none(float(np.mean(flat))),
        "p95": _round_or_none(float(np.percentile(flat, 95))),
        "median": _round_or_none(median),
        "peak_to_median": _round_or_none(peak / max(median, 1e-6)),
        "active_ratio": _round_or_none(float(np.mean(flat >= active_threshold)) if peak > 0 else 0.0),
    }


def _min_pair_distance(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    min_distance = None
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance = math.hypot(points[left][0] - points[right][0], points[left][1] - points[right][1])
            if min_distance is None or distance < min_distance:
                min_distance = distance
    return min_distance


def _top_detection_stats(detections) -> dict:
    detections = list(detections or [])
    if not detections:
        return {
            "detection_count": 0,
            "detection_top_score": None,
            "detection_top_rdi_peak": None,
            "detection_top_rai_peak": None,
            "detection_score_mean": None,
            "detection_min_separation_m": None,
        }
    points = [(float(item.x_m), float(item.y_m)) for item in detections]
    scores = [float(item.score) for item in detections]
    top = max(detections, key=lambda item: (float(item.score), float(item.rdi_peak), float(item.rai_peak)))
    return {
        "detection_count": len(detections),
        "detection_top_score": _round_or_none(top.score),
        "detection_top_rdi_peak": _round_or_none(top.rdi_peak),
        "detection_top_rai_peak": _round_or_none(top.rai_peak),
        "detection_score_mean": _round_or_none(float(np.mean(scores))),
        "detection_min_separation_m": _round_or_none(_min_pair_distance(points)),
    }


def _select_lead_track(confirmed_tracks):
    confirmed_tracks = list(confirmed_tracks or [])
    if not confirmed_tracks:
        return None
    primary = [track for track in confirmed_tracks if bool(getattr(track, "is_primary", False))]
    if primary:
        return primary[0]
    return max(confirmed_tracks, key=lambda track: (float(track.confidence), float(track.score), int(track.hits)))


def _track_quality_stats(confirmed_tracks, tentative_tracks) -> dict:
    confirmed_tracks = list(confirmed_tracks or [])
    tentative_tracks = list(tentative_tracks or [])
    all_tracks = confirmed_tracks + tentative_tracks
    residuals = [float(track.measurement_residual_m) for track in confirmed_tracks]
    qualities = [float(track.measurement_quality) for track in confirmed_tracks]
    lead = _select_lead_track(confirmed_tracks)
    result = {
        "confirmed_track_count": len(confirmed_tracks),
        "tentative_track_count": len(tentative_tracks),
        "active_track_count": len(all_tracks),
        "confirmed_residual_mean_m": _round_or_none(float(np.mean(residuals)) if residuals else None),
        "confirmed_residual_max_m": _round_or_none(float(np.max(residuals)) if residuals else None),
        "confirmed_quality_mean": _round_or_none(float(np.mean(qualities)) if qualities else None),
        "lead_track_id": None,
        "lead_x_m": None,
        "lead_y_m": None,
        "lead_range_m": None,
        "lead_angle_deg": None,
        "lead_confidence": None,
        "lead_measurement_quality": None,
        "lead_measurement_residual_m": None,
    }
    if lead is not None:
        result.update(
            {
                "lead_track_id": int(lead.track_id),
                "lead_x_m": _round_or_none(lead.x_m),
                "lead_y_m": _round_or_none(lead.y_m),
                "lead_range_m": _round_or_none(lead.range_m),
                "lead_angle_deg": _round_or_none(lead.angle_deg, digits=3),
                "lead_confidence": _round_or_none(lead.confidence),
                "lead_measurement_quality": _round_or_none(lead.measurement_quality),
                "lead_measurement_residual_m": _round_or_none(lead.measurement_residual_m),
            }
        )
    return result


def _slowest_stage(stage_timings_ms: dict) -> tuple[str | None, float | None]:
    if not stage_timings_ms:
        return None, None
    candidates = {
        key: float(value)
        for key, value in stage_timings_ms.items()
        if value is not None and key != "compute_total_ms"
    }
    if not candidates:
        return None, None
    name = max(candidates, key=candidates.get)
    return name, candidates[name]


def _frame_bottleneck(feature: dict) -> tuple[str, float, str]:
    if feature.get("invalid"):
        return "transport_frame", 10.0, "raw frame이 invalid로 표시되었습니다."
    compute_total = feature.get("compute_total_ms")
    if compute_total is not None and float(compute_total) >= 100.0:
        return "compute_over_budget", 9.0, f"compute_total_ms={float(compute_total):.1f}가 100ms budget을 넘었습니다."
    if compute_total is not None and float(compute_total) >= 80.0:
        return "compute_near_budget", 6.5, f"compute_total_ms={float(compute_total):.1f}가 budget에 가깝습니다."
    if int(feature.get("detection_count") or 0) == 0:
        return "detection_dropout", 8.0, "detection 후보가 0개입니다."
    if int(feature.get("tracker_input_count") or 0) > 0 and int(feature.get("confirmed_track_count") or 0) == 0:
        return "tracking_not_confirming", 7.0, "tracker input은 있지만 confirmed track이 없습니다."
    if bool(feature.get("lead_switch")):
        return "lead_id_switch", 7.0, "lead track id가 이전 프레임 대비 변경되었습니다."
    lead_step = feature.get("lead_step_m")
    if lead_step is not None and float(lead_step) >= 0.45:
        return "path_jump", 8.0, f"lead step={float(lead_step):.3f}m로 큽니다."
    residual = feature.get("lead_measurement_residual_m")
    if residual is not None and float(residual) >= 0.18:
        return "representative_point_jump", 7.5, f"lead residual={float(residual):.3f}m로 큽니다."
    candidate_count = int(feature.get("detection_count") or 0)
    confirmed_count = int(feature.get("confirmed_track_count") or 0)
    if confirmed_count > 0 and candidate_count / max(confirmed_count, 1) >= 2.0:
        return "detection_over_split", 6.0, "confirmed track 대비 detection 후보가 많습니다."
    rai_contrast = feature.get("rai_peak_to_median")
    if rai_contrast is not None and float(rai_contrast) <= 2.0:
        return "weak_rai_evidence", 5.5, "RAI peak contrast가 낮습니다."
    return "ok", 1.0, "강한 frame-level 병목이 보이지 않습니다."


def _build_frame_feature(processed_frame, artifacts, *, ordinal: int, previous_lead: dict | None) -> dict:
    rdi_stats = _array_quality_stats(artifacts.get("rdi"))
    rai_stats = _array_quality_stats(artifacts.get("rai"))
    detection_stats = _top_detection_stats(processed_frame.detections)
    track_stats = _track_quality_stats(processed_frame.confirmed_tracks, processed_frame.tentative_tracks)
    slowest_name, slowest_ms = _slowest_stage(processed_frame.stage_timings_ms)

    lead_id = track_stats.get("lead_track_id")
    lead_x = track_stats.get("lead_x_m")
    lead_y = track_stats.get("lead_y_m")
    lead_step = None
    lead_switch = False
    same_lead_id = None
    if previous_lead and lead_id is not None and lead_x is not None and lead_y is not None:
        same_lead_id = bool(int(lead_id) == int(previous_lead["lead_track_id"]))
        lead_switch = not same_lead_id
        lead_step = math.hypot(float(lead_x) - float(previous_lead["lead_x_m"]), float(lead_y) - float(previous_lead["lead_y_m"]))

    stage_timings = processed_frame.stage_timings_ms or {}
    feature = {
        "schema_version": STAGE_FEATURE_SCHEMA_VERSION,
        "ordinal": int(ordinal),
        "frame_id": int(processed_frame.frame_id),
        "capture_ts": _round_or_none(processed_frame.capture_ts, digits=6),
        "invalid": bool(processed_frame.invalid),
        "invalid_reason": processed_frame.invalid_reason,
        "udp_gap_count": int(processed_frame.udp_gap_count),
        "byte_mismatch_count": int(processed_frame.byte_mismatch_count),
        "out_of_sequence_count": int(processed_frame.out_of_sequence_count),
        "packets_in_frame": int(processed_frame.packets_in_frame),
        "tracker_policy": processed_frame.tracker_policy,
        "tracker_input_count": int(processed_frame.tracker_input_count),
        "track_birth_blocked": bool(processed_frame.track_birth_blocked),
        "rdi_max": rdi_stats["max"],
        "rdi_mean": rdi_stats["mean"],
        "rdi_p95": rdi_stats["p95"],
        "rdi_peak_to_median": rdi_stats["peak_to_median"],
        "rdi_active_ratio": rdi_stats["active_ratio"],
        "rai_max": rai_stats["max"],
        "rai_mean": rai_stats["mean"],
        "rai_p95": rai_stats["p95"],
        "rai_peak_to_median": rai_stats["peak_to_median"],
        "rai_active_ratio": rai_stats["active_ratio"],
        **detection_stats,
        **track_stats,
        "lead_step_m": _round_or_none(lead_step),
        "lead_switch": bool(lead_switch),
        "same_lead_id_as_previous": same_lead_id,
        "compute_total_ms": _round_or_none(stage_timings.get("compute_total_ms"), digits=3),
        "cube_ms": _round_or_none(stage_timings.get("cube_ms"), digits=3),
        "shared_fft2_ms": _round_or_none(stage_timings.get("shared_fft2_ms"), digits=3),
        "range_doppler_project_ms": _round_or_none(stage_timings.get("range_doppler_project_ms"), digits=3),
        "range_angle_project_ms": _round_or_none(stage_timings.get("range_angle_project_ms"), digits=3),
        "integrate_rdi_ms": _round_or_none(stage_timings.get("integrate_rdi_ms"), digits=3),
        "collapse_rai_ms": _round_or_none(stage_timings.get("collapse_rai_ms"), digits=3),
        "detect_ms": _round_or_none(stage_timings.get("detect_ms"), digits=3),
        "track_ms": _round_or_none(stage_timings.get("track_ms"), digits=3),
        "slowest_stage_name": slowest_name,
        "slowest_stage_ms": _round_or_none(slowest_ms, digits=3),
    }
    label, severity, evidence = _frame_bottleneck(feature)
    feature.update(
        {
            "frame_bottleneck": label,
            "frame_severity_10": severity,
            "frame_evidence": evidence,
        }
    )
    return feature


def _numeric_summary(features: list[dict], key: str) -> dict:
    values = [
        float(feature[key])
        for feature in features
        if feature.get(key) is not None and np.isfinite(float(feature[key]))
    ]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": _round_or_none(float(np.mean(array))),
        "p50": _round_or_none(float(np.percentile(array, 50))),
        "p95": _round_or_none(float(np.percentile(array, 95))),
        "max": _round_or_none(float(np.max(array))),
    }


def _build_feature_summary(features: list[dict]) -> dict:
    total = len(features)
    counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    for feature in features:
        label = str(feature.get("frame_bottleneck") or "unknown")
        counts[label] = counts.get(label, 0) + 1
        stage_name = str(feature.get("slowest_stage_name") or "unknown")
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1

    ok_count = counts.get("ok", 0)
    top_bottleneck = None
    if counts:
        top_bottleneck = max(counts.items(), key=lambda item: item[1])[0]
    return {
        "schema_version": STAGE_FEATURE_SCHEMA_VERSION,
        "generated_at": _now(),
        "frame_count": total,
        "ok_frame_count": ok_count,
        "ok_frame_rate": _round_or_none(ok_count / total if total else None),
        "invalid_frame_count": sum(1 for feature in features if feature.get("invalid")),
        "lead_switch_count": sum(1 for feature in features if feature.get("lead_switch")),
        "trackless_frame_count": sum(1 for feature in features if int(feature.get("confirmed_track_count") or 0) == 0),
        "top_frame_bottleneck": top_bottleneck,
        "frame_bottleneck_counts": [
            {
                "frame_bottleneck": label,
                "count": count,
                "probability": _round_or_none(count / total if total else None),
            }
            for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "slowest_stage_counts": [
            {
                "stage": label,
                "count": count,
                "probability": _round_or_none(count / total if total else None),
            }
            for label, count in sorted(stage_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "metrics": {
            "compute_total_ms": _numeric_summary(features, "compute_total_ms"),
            "detect_ms": _numeric_summary(features, "detect_ms"),
            "track_ms": _numeric_summary(features, "track_ms"),
            "lead_step_m": _numeric_summary(features, "lead_step_m"),
            "lead_measurement_residual_m": _numeric_summary(features, "lead_measurement_residual_m"),
            "detection_count": _numeric_summary(features, "detection_count"),
            "confirmed_track_count": _numeric_summary(features, "confirmed_track_count"),
            "rai_peak_to_median": _numeric_summary(features, "rai_peak_to_median"),
        },
    }


def _count_trace_stage_values(traces: list[dict], dotted_key: str) -> list[dict]:
    counts: dict[str, int] = {}
    for trace in traces:
        current = trace
        for key in dotted_key.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        label = str(current if current not in (None, "") else "none")
        counts[label] = counts.get(label, 0) + 1
    total = max(len(traces), 1)
    return [
        {
            "label": label,
            "count": count,
            "probability": _round_or_none(count / total),
        }
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _build_trace_summary(traces: list[dict]) -> dict:
    total = len(traces)
    stage_counts = []
    for trace in traces:
        detection = trace.get("detection") or {}
        tracker = trace.get("tracker") or {}
        stage_counts.append(
            {
                "cfar_candidates": int(_nested_get(detection, "cfar", "candidate_count", default=0) or 0),
                "angle_passed": int(_nested_get(detection, "angle_validation", "passed_count", default=0) or 0),
                "coarse_merge_after": int(_nested_get(detection, "candidate_merge_coarse", "after_count", default=0) or 0),
                "body_refined": int(_nested_get(detection, "body_center_refinement", "refined_count", default=0) or 0),
                "final_merge_after": int(_nested_get(detection, "candidate_merge_final", "after_count", default=0) or 0),
                "dbscan_output": int(_nested_get(detection, "dbscan", "output_count", default=0) or 0),
                "tracker_input": int(_nested_get(trace, "tracker_input_filter", "tracker_input_count", default=0) or 0),
                "association_matched": int(_nested_get(tracker, "association", "matched_count", default=0) or 0),
                "births": len(_nested_get(tracker, "track_lifecycle", "births", default=[]) or []),
                "deleted": len(_nested_get(tracker, "track_lifecycle", "deleted_track_ids", default=[]) or []),
                "display_confirmed": int(_nested_get(trace, "display_output", "confirmed_count", default=0) or 0),
            }
        )

    metrics = {}
    for key in [
        "cfar_candidates",
        "angle_passed",
        "coarse_merge_after",
        "body_refined",
        "final_merge_after",
        "dbscan_output",
        "tracker_input",
        "association_matched",
        "births",
        "deleted",
        "display_confirmed",
    ]:
        metrics[key] = _numeric_summary(stage_counts, key)

    return {
        "schema_version": 1,
        "generated_at": _now(),
        "frame_count": total,
        "metrics": metrics,
        "detection_early_exit_counts": _count_trace_stage_values(traces, "detection.early_exit"),
        "tracker_policy_counts": _count_trace_stage_values(traces, "tracker_input_filter.policy"),
    }


def build_stage_cache(
    project_root: Path,
    session_id: str,
    *,
    frame_limit: int | None = None,
    force: bool = False,
) -> dict:
    project_root = Path(project_root).resolve()
    run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        registry.refresh_registry(project_root)
        run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        raise FileNotFoundError(f"Run session not found in registry: {session_id}")

    capture_dir = _resolve_capture_dir(project_root, run_detail)
    capture_manifest, _, _ = load_raw_capture(capture_dir)
    runtime_summary = _merge_runtime_summary(run_detail, capture_manifest)
    components = _build_runtime_components(project_root, runtime_summary)

    paths = stage_cache_paths(project_root, session_id)
    requested_limit = int(frame_limit) if frame_limit not in (None, 0) else None
    if not force and paths["manifest_path"].exists() and paths["frames_path"].exists():
        existing_manifest = load_stage_cache_manifest(project_root, session_id) or {}
        feature_files_ready = (
            paths["features_path"].exists()
            and paths["feature_summary_path"].exists()
            and paths["trace_path"].exists()
            and paths["trace_summary_path"].exists()
            and int(existing_manifest.get("schema_version") or 0) >= STAGE_CACHE_SCHEMA_VERSION
        )
        if existing_manifest.get("frame_limit_requested") == requested_limit and feature_files_ready:
            return existing_manifest

    _clear_existing_cache(paths)

    processed_count = 0
    artifact_keys = ["cube_preview", "rdi", "rai"]
    frame_features: list[dict] = []
    frame_traces: list[dict] = []
    previous_lead: dict | None = None
    for ordinal, raw_frame in enumerate(iter_raw_capture_frame_packets(capture_dir)):
        if frame_limit is not None and int(frame_limit) > 0 and ordinal >= int(frame_limit):
            break

        processed_frame, artifacts = process_frame_packet(
            raw_frame,
            runtime_config=components["runtime_config"],
            detection_region=components["detection_region"],
            min_range_bin=components["min_range_bin"],
            max_range_bin=components["max_range_bin"],
            tracker=components["tracker"],
            block_track_birth_on_invalid=components["block_track_birth_on_invalid"],
            invalid_policy=components["invalid_policy"],
            detection_params=components["detection_params"],
            capture_stage_timing=True,
            return_artifacts=True,
            capture_trace=True,
        )
        artifact_file = paths["artifacts_dir"] / f"frame_{ordinal:06d}.npz"
        np.savez_compressed(
            artifact_file,
            cube_preview=np.asarray(artifacts["cube_preview"], dtype=np.float32),
            rdi=np.asarray(artifacts["rdi"], dtype=np.float32),
            rai=np.asarray(artifacts["rai"], dtype=np.float32),
        )
        frame_record = {
            "ordinal": int(ordinal),
            "frame_id": int(processed_frame.frame_id),
            "capture_ts": round(float(processed_frame.capture_ts), 6),
            "assembled_ts": round(float(processed_frame.assembled_ts), 6),
            "processed_ts": round(float(processed_frame.processed_ts or 0.0), 6),
            "invalid": bool(processed_frame.invalid),
            "invalid_reason": processed_frame.invalid_reason,
            "udp_gap_count": int(processed_frame.udp_gap_count),
            "byte_mismatch_count": int(processed_frame.byte_mismatch_count),
            "out_of_sequence_count": int(processed_frame.out_of_sequence_count),
            "packets_in_frame": int(processed_frame.packets_in_frame),
            "tracker_policy": processed_frame.tracker_policy,
            "tracker_input_count": int(processed_frame.tracker_input_count),
            "track_birth_blocked": bool(processed_frame.track_birth_blocked),
            "detections": [_serialize_detection(item) for item in processed_frame.detections],
            "tracker_input_detections": [
                _serialize_detection(item) for item in artifacts["tracker_input_detections"]
            ],
            "confirmed_tracks": [_serialize_track(item) for item in processed_frame.confirmed_tracks],
            "tentative_tracks": [_serialize_track(item) for item in processed_frame.tentative_tracks],
            "stage_timings_ms": processed_frame.stage_timings_ms,
            "artifact_file": str(artifact_file.relative_to(paths["cache_dir"])),
            "artifact_shapes": {
                "radar_cube_shape": list(artifacts["radar_cube_shape"]),
                "shared_fft_shape": list(artifacts["shared_fft_shape"]),
                "rdi_cube_shape": list(artifacts["rdi_cube_shape"]),
                "rai_cube_shape": list(artifacts["rai_cube_shape"]),
                "cube_preview_shape": list(np.asarray(artifacts["cube_preview"]).shape),
                "rdi_shape": list(np.asarray(artifacts["rdi"]).shape),
                "rai_shape": list(np.asarray(artifacts["rai"]).shape),
            },
        }
        _append_jsonl(paths["frames_path"], frame_record)

        feature_record = _build_frame_feature(
            processed_frame,
            artifacts,
            ordinal=ordinal,
            previous_lead=previous_lead,
        )
        if feature_record.get("lead_track_id") is not None:
            previous_lead = {
                "lead_track_id": int(feature_record["lead_track_id"]),
                "lead_x_m": float(feature_record["lead_x_m"]),
                "lead_y_m": float(feature_record["lead_y_m"]),
            }
        _append_jsonl(paths["features_path"], feature_record)
        frame_features.append(feature_record)
        frame_trace = artifacts.get("frame_trace") or {}
        _append_jsonl(paths["trace_path"], frame_trace)
        frame_traces.append(frame_trace)
        processed_count += 1

    feature_summary = _build_feature_summary(frame_features)
    _write_json(paths["feature_summary_path"], feature_summary)
    trace_summary = _build_trace_summary(frame_traces)
    _write_json(paths["trace_summary_path"], trace_summary)

    manifest = {
        "schema_version": STAGE_CACHE_SCHEMA_VERSION,
        "session_id": str(session_id),
        "source_session_dir": run_detail.get("session_dir"),
        "capture_id": run_detail.get("capture_id"),
        "capture_dir": str(capture_dir),
        "generated_at": _now(),
        "frame_count": int(processed_count),
        "frame_limit_requested": requested_limit,
        "artifact_keys": artifact_keys,
        "feature_keys": [
            "raw_health",
            "rdi_quality",
            "rai_quality",
            "detection_counts",
            "tracker_counts",
            "lead_track_continuity",
            "stage_timings",
            "frame_bottleneck",
        ],
        "trace_keys": [
            "raw_udp_packets",
            "frame_parsing",
            "radar_cube",
            "static_removal",
            "shared_fft",
            "rdi",
            "rai",
            "detection.cfar",
            "detection.angle_validation",
            "detection.body_center_refinement",
            "detection.candidate_merge_final",
            "detection.dbscan",
            "tracker_input_filter",
            "tracker.kalman_prediction",
            "tracker.association",
            "tracker.kalman_update",
            "tracker.track_lifecycle",
            "display_output",
        ],
        "notes": [
            "Stage cache v2 stores cube preview + RDI + RAI heatmaps, serialized detections/tracks, frame_features.jsonl, and frame_trace.jsonl.",
            "Full 3D radar cube arrays are not stored in trace; compact stats/top-K candidates are stored for offline diagnostics.",
        ],
        "feature_summary": feature_summary,
        "trace_summary": trace_summary,
        "runtime": {
            "cfg_path": str(components["cfg_path"]),
            "remove_static": bool(components["runtime_config"].remove_static),
            "doppler_guard_bins": int(components["runtime_config"].doppler_guard_bins),
            "range_resolution_m": round(float(components["runtime_config"].range_resolution_m), 6),
            "max_range_m": round(float(components["runtime_config"].max_range_m), 4),
            "range_fft_size": int(components["runtime_config"].range_fft_size),
            "doppler_fft_size": int(components["runtime_config"].doppler_fft_size),
            "angle_fft_size": int(components["runtime_config"].angle_fft_size),
            "lateral_axis_sign": float(components["runtime_config"].lateral_axis_sign),
        },
        "roi": {
            "lateral_m": round(float(components["roi_lateral_m"]), 4),
            "forward_m": round(float(components["roi_forward_m"]), 4),
            "min_forward_m": round(float(components["roi_min_forward_m"]), 4),
            "min_range_bin": int(components["min_range_bin"]),
            "max_range_bin": int(components["max_range_bin"]),
        },
    }
    _write_json(paths["manifest_path"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stage-wise replay cache for a radar run.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--session", required=True, help="Run session id to build the stage cache for.")
    parser.add_argument("--limit", type=int, default=0, help="Optional frame limit. 0 means all frames.")
    parser.add_argument("--force", action="store_true", help="Rebuild cache even if it already exists.")
    args = parser.parse_args()

    manifest = build_stage_cache(
        Path(args.project_root),
        args.session,
        frame_limit=(args.limit or None),
        force=bool(args.force),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
