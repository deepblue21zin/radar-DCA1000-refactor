from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import inspect
import json
import math
from pathlib import Path

import numpy as np

from tools.lab import registry
from tools.runtime_core.detection import DetectionRegion, detect_targets as _RuntimeDetectTargets
from tools.runtime_core.radar_runtime import parse_runtime_config, radial_bin_limit
from tools.runtime_core.real_time_process import (
    _serialize_detection,
    _serialize_track,
    iter_raw_capture_frame_packets,
    load_raw_capture,
    process_frame_packet as _runtime_process_frame_packet,
)
from tools.runtime_core.runtime_settings import load_runtime_settings
from tools.runtime_core.tracking import MultiTargetTracker as _RuntimeMultiTargetTracker


STAGE_CACHE_SCHEMA_VERSION = 7
STAGE_FEATURE_SCHEMA_VERSION = 1


def _filter_tracker_kwargs_for_loaded_signature(kwargs: dict) -> dict:
    parameters = inspect.signature(_RuntimeMultiTargetTracker.__init__).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    supported = set(parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in supported}


def MultiTargetTracker(*args, **kwargs):
    if args:
        return _RuntimeMultiTargetTracker(*args, **kwargs)
    return _RuntimeMultiTargetTracker(**_filter_tracker_kwargs_for_loaded_signature(kwargs))


class _NoOpTracker:
    def update(self, *args, **kwargs):
        return [], []


def _filter_process_frame_kwargs_for_loaded_signature(kwargs: dict) -> dict:
    parameters = inspect.signature(_runtime_process_frame_packet).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    supported = set(parameters) - {"raw_frame"}
    return {key: value for key, value in kwargs.items() if key in supported}


def process_frame_packet(raw_frame, **kwargs):
    parameters = inspect.signature(_runtime_process_frame_packet).parameters
    tracker_enabled = bool(kwargs.get("tracker_enabled", True))
    if "tracker_enabled" not in parameters and not tracker_enabled:
        kwargs["tracker"] = _NoOpTracker()
    return _runtime_process_frame_packet(
        raw_frame,
        **_filter_process_frame_kwargs_for_loaded_signature(kwargs),
    )


def _filter_detection_params_for_loaded_signature(params: dict) -> dict:
    parameters = inspect.signature(_RuntimeDetectTargets).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return params
    supported = set(parameters) - {
        "rdi_map",
        "rai_map",
        "runtime_config",
        "min_range_bin",
        "max_range_bin",
        "detection_region",
    }
    return {key: value for key, value in params.items() if key in supported}


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


def _source_capture_candidates(project_root: Path, value: str | Path | None) -> list[Path]:
    direct = _as_path(project_root, value)
    candidates = []
    if direct is not None:
        candidates.append(direct)
    if not value:
        return candidates

    session_id = Path(str(value)).name
    for live_session_dir in (
        project_root / "logs" / "live_motion_viewer" / str(value),
        project_root / "logs" / "live_motion_viewer" / session_id,
    ):
        summary_path = live_session_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        source_capture = (
            _nested_get(summary, "session_meta", "source_capture")
            or summary.get("source_capture")
            or _nested_get(summary, "runtime_config", "log_source_capture")
        )
        nested = _as_path(project_root, source_capture)
        if nested is not None:
            candidates.append(nested)
    return candidates


RAW_CAPTURE_REQUIRED_FILES = ("capture_manifest.json", "raw_frames_index.jsonl", "raw_frames.i16")


def _raw_capture_missing_files(capture_dir: Path) -> list[str]:
    missing: list[str] = []
    for filename in RAW_CAPTURE_REQUIRED_FILES:
        file_path = capture_dir / filename
        if not file_path.exists():
            missing.append(filename)
            continue
        if filename != "capture_manifest.json" and file_path.stat().st_size <= 0:
            missing.append(f"{filename} (empty)")
    return missing


def _capture_candidate_paths(project_root: Path, run_detail: dict) -> list[Path]:
    capture_candidates = []
    if run_detail.get("capture_id"):
        capture_candidates.extend(
            _source_capture_candidates(project_root, project_root / "logs" / "raw" / str(run_detail["capture_id"]))
        )

    session_meta = run_detail.get("session_meta") or {}
    runtime_summary = run_detail.get("runtime_config") or {}
    summary = run_detail.get("summary") or {}

    capture_candidates.extend(
        value
        for value in [
            _as_path(project_root, session_meta.get("raw_capture_dir")),
            *(_source_capture_candidates(project_root, session_meta.get("source_capture"))),
            _as_path(project_root, runtime_summary.get("raw_capture_dir")),
            *(_source_capture_candidates(project_root, runtime_summary.get("log_source_capture"))),
            _as_path(project_root, _nested_get(summary, "session_meta", "raw_capture_dir")),
            *(_source_capture_candidates(project_root, _nested_get(summary, "session_meta", "source_capture"))),
            _as_path(project_root, _nested_get(summary, "runtime_config", "raw_capture_dir")),
            *(_source_capture_candidates(project_root, _nested_get(summary, "runtime_config", "log_source_capture"))),
        ]
        if value is not None
    )

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in capture_candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def raw_capture_status_for_run(project_root: Path, run_detail: dict) -> dict:
    checked: list[dict] = []
    for candidate in _capture_candidate_paths(project_root, run_detail):
        if not candidate.exists():
            checked.append(
                {
                    "path": str(candidate),
                    "usable": False,
                    "status": "missing_directory",
                    "missing": ["directory"],
                }
            )
            continue
        missing = _raw_capture_missing_files(candidate)
        if missing:
            checked.append(
                {
                    "path": str(candidate),
                    "usable": False,
                    "status": "incomplete_raw_capture",
                    "missing": missing,
                }
            )
            continue
        return {
            "usable": True,
            "status": "ready",
            "path": str(candidate),
            "missing": [],
            "checked": checked
            + [
                {
                    "path": str(candidate),
                    "usable": True,
                    "status": "ready",
                    "missing": [],
                }
            ],
        }

    return {
        "usable": False,
        "status": "no_usable_raw_capture",
        "path": None,
        "missing": [],
        "checked": checked,
    }


def _format_capture_status_error(status: dict) -> str:
    checked = status.get("checked") or []
    if not checked:
        return (
            "This run does not have a resolvable raw capture directory. "
            "Record a session with raw capture enabled or choose a raw-linked replay run."
        )
    details = []
    for item in checked[:5]:
        missing = ", ".join(item.get("missing") or [])
        suffix = f" missing={missing}" if missing else ""
        details.append(f"{item.get('path')} [{item.get('status')}]{suffix}")
    if len(checked) > 5:
        details.append(f"... {len(checked) - 5} more candidate(s)")
    return (
        "No usable replay raw capture was found. Stage Cache needs "
        "`capture_manifest.json`, `raw_frames_index.jsonl`, and `raw_frames.i16`. "
        "Checked: "
        + " | ".join(details)
    )


def _resolve_capture_dir(project_root: Path, run_detail: dict) -> Path:
    status = raw_capture_status_for_run(project_root, run_detail)
    if status.get("usable") and status.get("path"):
        return Path(status["path"])

    raise FileNotFoundError(_format_capture_status_error(status))



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


def _bool_from_config(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _current_lateral_axis_sign(project_root: Path) -> tuple[float | None, str | None]:
    try:
        settings = load_runtime_settings(project_root)
    except Exception:
        return None, None

    invert = _nested_get(settings, "tuning", "processing", "invert_lateral_axis")
    if invert is None:
        return None, None
    return (-1.0 if _bool_from_config(invert) else 1.0), "current_tuning"


def _logged_lateral_axis_sign(runtime_summary: dict) -> tuple[float, str]:
    lateral_axis_sign = runtime_summary.get("lateral_axis_sign")
    if lateral_axis_sign is not None:
        return float(lateral_axis_sign), "logged_summary"

    invert = runtime_summary.get("invert_lateral_axis")
    if invert is None:
        invert = _nested_get(runtime_summary, "tuning_snapshot", "processing", "invert_lateral_axis", default=False)
    return (-1.0 if _bool_from_config(invert) else 1.0), "logged_tuning"


def _current_runtime_settings(project_root: Path) -> dict:
    try:
        return load_runtime_settings(project_root)
    except Exception:
        return {}


def _merge_mapping(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def _overlay_current_tuning(runtime_summary: dict, current_settings: dict) -> dict:
    current_tuning = _nested_get(current_settings, "tuning", default={}) or {}
    if not current_tuning:
        return dict(runtime_summary or {})

    summary = dict(runtime_summary or {})
    summary["tuning_snapshot"] = _merge_mapping(summary.get("tuning_snapshot") or {}, current_tuning)

    processing = current_tuning.get("processing") or {}
    for key in (
        "remove_static",
        "doppler_guard_bins",
        "angle_projection",
        "angle_phase_sign",
        "angle_source",
    ):
        if key in processing:
            summary[key] = processing[key]
    calibration = processing.get("channel_calibration") or {}
    if "enabled" in calibration:
        summary["channel_calibration_enabled"] = calibration["enabled"]
    tdm_compensation = processing.get("tdm_mimo_doppler_compensation") or {}
    if "enabled" in tdm_compensation:
        summary["tdm_mimo_doppler_compensation_enabled"] = tdm_compensation["enabled"]
    if "phase_sign" in tdm_compensation:
        summary["tdm_mimo_doppler_compensation_phase_sign"] = tdm_compensation["phase_sign"]
    if "slot_time_model" in tdm_compensation:
        summary["tdm_mimo_doppler_compensation_slot_time_model"] = tdm_compensation[
            "slot_time_model"
        ]
    if "reference_tx_slot" in tdm_compensation:
        summary["tdm_mimo_doppler_compensation_reference_tx_slot"] = tdm_compensation[
            "reference_tx_slot"
        ]

    roi = current_tuning.get("roi") or {}
    roi_key_map = {
        "lateral_m": "roi_lateral_m",
        "forward_m": "roi_forward_m",
        "min_forward_m": "roi_min_forward_m",
    }
    for source_key, summary_key in roi_key_map.items():
        if source_key in roi:
            summary[summary_key] = roi[source_key]

    detection = current_tuning.get("detection") or {}
    detection_key_map = {
        "max_targets": "max_targets",
        "allow_strongest_fallback": "allow_strongest_fallback",
        "dbscan_adaptive_eps_bands": "dbscan_adaptive_eps_bands",
        "cluster_min_samples": "cluster_min_samples",
        "cluster_velocity_weight": "cluster_velocity_weight",
    }
    for source_key, summary_key in detection_key_map.items():
        if source_key in detection:
            summary[summary_key] = detection[source_key]
    if isinstance(detection.get("algorithm"), dict):
        summary["detection_algorithm"] = detection["algorithm"]

    for key, value in (current_tuning.get("tracking") or {}).items():
        summary[f"track_{key}"] = value

    return summary


def _normalize_ablation_mode(mode: str | None) -> str:
    mode = str(mode or "baseline").strip().lower()
    allowed = {
        "baseline",
        "doppler_slice_angle",
        "rda_candidates",
        "person_blob",
        "blob_center",
        "person_aware_merge",
        "multi_tracker_relaxed",
        "no_body_center",
        "no_duplicate_suppression",
        "no_merge",
        "no_dbscan",
        "tracker_off",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported stage cache ablation mode: {mode}")
    return mode


def _tracking_value(runtime_summary: dict, key: str, default=None):
    return runtime_summary.get(
        f"track_{key}",
        _nested_get(runtime_summary, "tuning_snapshot", "tracking", key, default=default),
    )


def _build_runtime_components(
    project_root: Path,
    runtime_summary: dict,
    *,
    ablation_mode: str = "baseline",
    cfg_path_override: str | Path | None = None,
    angle_phase_sign_override: float | None = None,
    lateral_axis_sign_override: float | None = None,
    tdm_compensation_override: bool | None = None,
    tdm_phase_sign_override: float | None = None,
):
    ablation_mode = _normalize_ablation_mode(ablation_mode)
    current_settings = _current_runtime_settings(project_root)
    runtime_summary = _overlay_current_tuning(runtime_summary, current_settings)
    current_processing = _nested_get(current_settings, "tuning", "processing", default={}) or {}
    if cfg_path_override is not None:
        cfg_path = _as_path(project_root, cfg_path_override)
        if cfg_path is None or not cfg_path.exists():
            raise FileNotFoundError(f"Could not resolve override cfg path: {cfg_path_override}")
        cfg_path_source = "override"
    else:
        cfg_path = _resolve_cfg_path(project_root, runtime_summary)
        cfg_path_source = "runtime_summary"
    remove_static = bool(
        runtime_summary.get(
            "remove_static",
            _nested_get(runtime_summary, "tuning_snapshot", "processing", "remove_static", default=True),
        )
    )
    doppler_guard_bins = int(
        runtime_summary.get(
            "doppler_guard_bins",
            _nested_get(runtime_summary, "tuning_snapshot", "processing", "doppler_guard_bins", default=1),
        )
    )
    angle_projection = runtime_summary.get(
        "angle_projection",
        _nested_get(runtime_summary, "tuning_snapshot", "processing", "angle_projection", default="fft1d"),
    )
    current_angle_phase_sign = current_processing.get("angle_phase_sign")
    if angle_phase_sign_override is not None:
        angle_phase_sign = angle_phase_sign_override
        angle_phase_sign_source = "override"
    elif current_angle_phase_sign is not None:
        angle_phase_sign = current_angle_phase_sign
        angle_phase_sign_source = "current_tuning"
    else:
        angle_phase_sign = runtime_summary.get(
            "angle_phase_sign",
            _nested_get(runtime_summary, "tuning_snapshot", "processing", "angle_phase_sign", default=-1.0),
        )
        angle_phase_sign_source = "logged_summary"
    angle_source = runtime_summary.get(
        "angle_source",
        _nested_get(
            runtime_summary,
            "tuning_snapshot",
            "processing",
            "angle_source",
            default=current_processing.get("angle_source", "collapsed_rai"),
        ),
    )
    channel_calibration = _nested_get(
        runtime_summary,
        "tuning_snapshot",
        "processing",
        "channel_calibration",
        default=current_processing.get("channel_calibration", {}),
    ) or {}
    channel_calibration_enabled = runtime_summary.get(
        "channel_calibration_enabled",
        channel_calibration.get("enabled", False),
    )
    channel_calibration_coefficients = channel_calibration.get("coefficients", [])
    tdm_compensation = _nested_get(
        runtime_summary,
        "tuning_snapshot",
        "processing",
        "tdm_mimo_doppler_compensation",
        default=current_processing.get("tdm_mimo_doppler_compensation", {}),
    ) or {}
    if tdm_compensation_override is not None:
        tdm_compensation_enabled = bool(tdm_compensation_override)
        tdm_compensation_source = "override"
    else:
        tdm_compensation_enabled = bool(
            runtime_summary.get(
                "tdm_mimo_doppler_compensation_enabled",
                tdm_compensation.get("enabled", False),
            )
        )
        tdm_compensation_source = (
            "current_tuning"
            if "tdm_mimo_doppler_compensation" in current_processing
            else "logged_summary"
        )
    if tdm_phase_sign_override is not None:
        tdm_compensation_phase_sign = float(tdm_phase_sign_override)
        tdm_phase_sign_source = "override"
    else:
        tdm_compensation_phase_sign = float(
            runtime_summary.get(
                "tdm_mimo_doppler_compensation_phase_sign",
                tdm_compensation.get("phase_sign", 1.0),
            )
        )
        tdm_phase_sign_source = tdm_compensation_source
    tdm_compensation_slot_time_model = str(
        runtime_summary.get(
            "tdm_mimo_doppler_compensation_slot_time_model",
            tdm_compensation.get("slot_time_model", "uniform_tx_slot"),
        )
    ).strip().lower()
    tdm_compensation_reference_tx_slot = int(
        runtime_summary.get(
            "tdm_mimo_doppler_compensation_reference_tx_slot",
            tdm_compensation.get("reference_tx_slot", 0),
        )
    )
    angle_elevation_min_deg = _nested_get(
        runtime_summary,
        "tuning_snapshot",
        "processing",
        "angle_elevation_min_deg",
        default=-40.0,
    )
    angle_elevation_max_deg = _nested_get(
        runtime_summary,
        "tuning_snapshot",
        "processing",
        "angle_elevation_max_deg",
        default=40.0,
    )
    angle_elevation_step_deg = _nested_get(
        runtime_summary,
        "tuning_snapshot",
        "processing",
        "angle_elevation_step_deg",
        default=4.0,
    )
    if lateral_axis_sign_override is not None:
        lateral_axis_sign = lateral_axis_sign_override
        lateral_axis_sign_source = "override"
    else:
        lateral_axis_sign, lateral_axis_sign_source = _logged_lateral_axis_sign(runtime_summary)
    if lateral_axis_sign is None:
        lateral_axis_sign, lateral_axis_sign_source = _current_lateral_axis_sign(project_root)

    runtime_config = parse_runtime_config(
        cfg_path,
        remove_static=remove_static,
        doppler_guard_bins=doppler_guard_bins,
        lateral_axis_sign=float(lateral_axis_sign),
    )
    runtime_config_updates = {}
    if hasattr(runtime_config, "angle_projection"):
        runtime_config_updates["angle_projection"] = str(angle_projection or "fft1d").strip().lower()
    if hasattr(runtime_config, "angle_elevation_min_deg"):
        runtime_config_updates["angle_elevation_min_deg"] = float(angle_elevation_min_deg)
    if hasattr(runtime_config, "angle_elevation_max_deg"):
        runtime_config_updates["angle_elevation_max_deg"] = float(angle_elevation_max_deg)
    if hasattr(runtime_config, "angle_elevation_step_deg"):
        runtime_config_updates["angle_elevation_step_deg"] = float(angle_elevation_step_deg)
    if hasattr(runtime_config, "angle_phase_sign"):
        runtime_config_updates["angle_phase_sign"] = float(angle_phase_sign)
    if hasattr(runtime_config, "angle_source"):
        runtime_config_updates["angle_source"] = str(angle_source or "collapsed_rai").strip().lower()
    if hasattr(runtime_config, "channel_calibration_enabled"):
        runtime_config_updates["channel_calibration_enabled"] = bool(channel_calibration_enabled)
    if hasattr(runtime_config, "channel_calibration_coefficients"):
        from tools.runtime_core.radar_runtime import _parse_complex_coefficients

        runtime_config_updates["channel_calibration_coefficients"] = _parse_complex_coefficients(
            channel_calibration_coefficients
        )
    if hasattr(runtime_config, "tdm_mimo_doppler_compensation_enabled"):
        runtime_config_updates["tdm_mimo_doppler_compensation_enabled"] = bool(
            tdm_compensation_enabled
        )
    if hasattr(runtime_config, "tdm_mimo_doppler_compensation_phase_sign"):
        runtime_config_updates["tdm_mimo_doppler_compensation_phase_sign"] = float(
            tdm_compensation_phase_sign
        )
    if hasattr(runtime_config, "tdm_mimo_doppler_compensation_slot_time_model"):
        runtime_config_updates["tdm_mimo_doppler_compensation_slot_time_model"] = str(
            tdm_compensation_slot_time_model or "uniform_tx_slot"
        ).strip().lower()
    if hasattr(runtime_config, "tdm_mimo_doppler_compensation_reference_tx_slot"):
        runtime_config_updates["tdm_mimo_doppler_compensation_reference_tx_slot"] = int(
            tdm_compensation_reference_tx_slot
        )
    if runtime_config_updates:
        runtime_config = replace(runtime_config, **runtime_config_updates)

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
    resolved_angle_source = str(
        getattr(runtime_config, "angle_source", angle_source or "collapsed_rai") or "collapsed_rai"
    ).strip().lower()
    detection_params["angle_source"] = resolved_angle_source
    detection_params.setdefault("range_doppler_ambiguity_suppression_enabled", True)
    detection_params.setdefault("range_doppler_ambiguity_range_tolerance_m", 0.32)
    detection_params.setdefault("range_doppler_ambiguity_doppler_bins", 2)
    detection_params.setdefault("range_doppler_ambiguity_min_angle_delta_deg", 14.0)
    detection_params.setdefault("range_doppler_ambiguity_min_separation_m", 0.70)
    detection_params.setdefault("range_doppler_ambiguity_mirror_x_tolerance_m", 0.50)
    detection_params.setdefault("range_doppler_ambiguity_mirror_y_tolerance_m", 0.45)
    detection_params.setdefault("range_doppler_ambiguity_min_abs_angle_deg", 4.0)
    detection_params.setdefault("rd_cfar_output_guard_enabled", True)
    detection_params.setdefault("rd_cfar_output_guard_max_references", 16)
    detection_params.setdefault("rd_cfar_output_guard_min_reference_score_ratio", 0.05)
    detection_params.setdefault("rd_cfar_output_guard_range_tolerance_m", 0.50)
    detection_params.setdefault("rd_cfar_output_guard_doppler_bins", 3)
    detection_params.setdefault("rd_cfar_output_guard_prefer_references", True)
    detection_params.setdefault("rd_cfar_output_guard_replace_shifted", True)
    detection_params.setdefault("rd_cfar_output_guard_replace_shift_m", 0.35)
    detection_params.setdefault("rd_cfar_output_guard_fallback_to_references", True)

    tracker_enabled = True
    tracker_overrides = {}
    if ablation_mode == "doppler_slice_angle":
        detection_params["angle_source"] = "doppler_slice_rai"
    elif ablation_mode == "rda_candidates":
        detection_params["angle_source"] = "doppler_slice_rai"
        detection_params["enable_dbscan"] = False
    elif ablation_mode == "person_blob":
        detection_params["angle_source"] = "doppler_slice_rai"
        detection_params["protect_multi_object_candidates"] = True
        detection_params["limit_output_to_object_count"] = True
        detection_params["person_blob_refinement_enabled"] = True
        detection_params["person_blob_doppler_radius_bins"] = 2
        detection_params["person_blob_min_points"] = 4
        detection_params["person_blob_floor_quantile"] = 0.68
        detection_params["person_blob_center_method"] = "weighted_median"
        detection_params["person_blob_peak_blend"] = 0.12
    elif ablation_mode == "blob_center":
        detection_params["angle_source"] = "doppler_slice_rai"
        detection_params["blob_center_refinement_enabled"] = True
        detection_params["blob_center_max_candidates"] = 36
        detection_params["blob_center_min_points"] = 4
        detection_params["blob_center_min_score_ratio"] = 0.04
        detection_params["blob_center_cluster_radius_m"] = 0.55
        detection_params["blob_center_cluster_radius_range_scale"] = 0.04
        detection_params["blob_center_cluster_radius_bands"] = [
            {"r_min": 0.0, "r_max": 1.5, "radius_m": 0.45},
            {"r_min": 1.5, "r_max": 3.0, "radius_m": 0.58},
            {"r_min": 3.0, "r_max": None, "radius_m": 0.72},
        ]
        detection_params["blob_center_doppler_radius_bins"] = 4
        detection_params["blob_center_method"] = "weighted_median_trimmed"
        detection_params["blob_center_trim_radius_m"] = 0.65
        detection_params["blob_center_floor_quantile"] = 0.65
        detection_params["blob_center_peak_blend"] = 0.0
        detection_params["blob_center_single_min_score_ratio"] = 0.12
        detection_params["blob_center_single_range_window_m"] = 1.05
        detection_params["blob_center_single_side_deadband_m"] = 0.15
        detection_params["blob_center_cube_range_radius_m"] = 0.45
        detection_params["blob_center_cube_angle_radius_deg"] = 12.0
        detection_params["blob_center_cube_relative_floor"] = 0.38
        detection_params["blob_center_dense_enabled"] = True
        detection_params["blob_center_dense_quantile"] = 0.995
        detection_params["blob_center_dense_min_normalized_power"] = 0.08
        detection_params["blob_center_dense_max_points"] = 2400
        detection_params["blob_center_dense_min_points"] = 8
        detection_params["blob_center_dense_grouping_mode"] = "rd_primary"
        detection_params["blob_center_dense_angle_radius_deg"] = 12.0
        detection_params["blob_center_dense_angle_floor_quantile"] = 0.70
        detection_params["blob_center_dense_angle_relative_floor"] = 0.38
        detection_params["blob_center_anchor_max_shift_m"] = 0.38
        detection_params["blob_center_anchor_blend"] = 0.82
        detection_params["range_doppler_ambiguity_suppression_enabled"] = True
        detection_params["range_doppler_ambiguity_range_tolerance_m"] = 0.32
        detection_params["range_doppler_ambiguity_doppler_bins"] = 2
        detection_params["range_doppler_ambiguity_min_angle_delta_deg"] = 14.0
        detection_params["range_doppler_ambiguity_min_separation_m"] = 0.70
        detection_params["range_doppler_ambiguity_mirror_x_tolerance_m"] = 0.50
        detection_params["range_doppler_ambiguity_mirror_y_tolerance_m"] = 0.45
        detection_params["range_doppler_ambiguity_min_abs_angle_deg"] = 4.0
        detection_params["rd_cfar_output_guard_enabled"] = True
        detection_params["rd_cfar_output_guard_max_references"] = 16
        detection_params["rd_cfar_output_guard_min_reference_score_ratio"] = 0.05
        detection_params["rd_cfar_output_guard_range_tolerance_m"] = 0.50
        detection_params["rd_cfar_output_guard_doppler_bins"] = 3
        detection_params["rd_cfar_output_guard_prefer_references"] = True
        detection_params["rd_cfar_output_guard_replace_shifted"] = True
        detection_params["rd_cfar_output_guard_replace_shift_m"] = 0.35
        detection_params["rd_cfar_output_guard_fallback_to_references"] = True
        detection_params["min_output_score"] = max(float(detection_params.get("min_output_score", 0.0) or 0.0), 0.25)
    elif ablation_mode == "person_aware_merge":
        detection_params["angle_source"] = "doppler_slice_rai"
        detection_params["protect_multi_object_candidates"] = True
    elif ablation_mode == "multi_tracker_relaxed":
        detection_params["angle_source"] = "doppler_slice_rai"
        detection_params["protect_multi_object_candidates"] = True
        detection_params["limit_output_to_object_count"] = True
        tracker_overrides = {
            "track_confirm_hits": 2,
            "track_birth_suppression_radius_m": 0.35,
            "track_primary_track_birth_scale": 1.0,
            "track_birth_suppression_weak_radius_scale": 1.0,
            "track_birth_suppression_score_ratio": 0.0,
            "track_birth_suppression_confidence_ratio": 0.0,
            "track_tentative_gate_factor": 1.0,
        }
    elif ablation_mode == "no_body_center":
        detection_params["enable_body_center_refinement"] = False
    elif ablation_mode == "no_duplicate_suppression":
        detection_params["duplicate_suppression_enabled"] = False
    elif ablation_mode == "no_merge":
        detection_params["enable_candidate_merge"] = False
    elif ablation_mode == "no_dbscan":
        detection_params["enable_dbscan"] = False
    elif ablation_mode == "tracker_off":
        tracker_enabled = False
    detection_params = _filter_detection_params_for_loaded_signature(detection_params)

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
        lateral_measurement_scale=float(runtime_summary.get("track_lateral_measurement_scale", 1.0)),
        forward_measurement_scale=float(runtime_summary.get("track_forward_measurement_scale", 1.0)),
        angle_resolution_rad=angle_resolution_rad,
        association_gate=float(runtime_summary.get("track_association_gate", 5.99)),
        doppler_center_bin=int(runtime_config.doppler_fft_size // 2),
        doppler_zero_guard_bins=int(runtime_summary.get("track_doppler_zero_guard_bins", 2)),
        doppler_gate_bins=int(runtime_summary.get("track_doppler_gate_bins", 0)),
        doppler_cost_weight=float(runtime_summary.get("track_doppler_cost_weight", 0.0)),
        min_confirmed_hits=int(tracker_overrides.get("track_confirm_hits", runtime_summary.get("track_confirm_hits", 2))),
        max_missed_frames=int(runtime_summary.get("track_max_misses", 8)),
        report_miss_tolerance=int(runtime_summary.get("track_report_miss_tolerance", 2)),
        lost_gate_factor=float(runtime_summary.get("track_lost_gate_factor", 1.2)),
        tentative_gate_factor=float(
            tracker_overrides.get("track_tentative_gate_factor", runtime_summary.get("track_tentative_gate_factor", 0.5))
        ),
        birth_suppression_radius_m=float(
            tracker_overrides.get("track_birth_suppression_radius_m", runtime_summary.get("track_birth_suppression_radius_m", 0.0))
        ),
        primary_track_birth_scale=float(
            tracker_overrides.get("track_primary_track_birth_scale", runtime_summary.get("track_primary_track_birth_scale", 1.0))
        ),
        birth_suppression_weak_radius_scale=float(
            tracker_overrides.get(
                "track_birth_suppression_weak_radius_scale",
                runtime_summary.get("track_birth_suppression_weak_radius_scale", 1.0),
            )
        ),
        birth_suppression_score_ratio=float(
            tracker_overrides.get(
                "track_birth_suppression_score_ratio",
                runtime_summary.get("track_birth_suppression_score_ratio", 0.0),
            )
        ),
        birth_suppression_confidence_ratio=float(
            tracker_overrides.get(
                "track_birth_suppression_confidence_ratio",
                runtime_summary.get("track_birth_suppression_confidence_ratio", 0.0),
            )
        ),
        birth_suppression_doppler_bins=int(
            runtime_summary.get("track_birth_suppression_doppler_bins", 0)
        ),
        birth_suppression_miss_tolerance=int(
            runtime_summary.get("track_birth_suppression_miss_tolerance", 0)
        ),
        primary_track_hold_frames=int(runtime_summary.get("track_primary_track_hold_frames", 0)),
        lateral_deadband_m=float(runtime_summary.get("track_lateral_deadband_m", 0.0)),
        lateral_deadband_range_scale=float(runtime_summary.get("track_lateral_deadband_range_scale", 0.0)),
        lateral_smoothing_alpha=float(runtime_summary.get("track_lateral_smoothing_alpha", 1.0)),
        lateral_velocity_damping=float(runtime_summary.get("track_lateral_velocity_damping", 1.0)),
        lateral_range_damping_enabled=bool(
            runtime_summary.get("track_lateral_range_damping_enabled", False)
        ),
        lateral_range_damping_start_m=float(
            runtime_summary.get("track_lateral_range_damping_start_m", 1.4)
        ),
        lateral_range_damping_full_m=float(
            runtime_summary.get("track_lateral_range_damping_full_m", 3.8)
        ),
        lateral_range_damping_min_alpha=float(
            runtime_summary.get("track_lateral_range_damping_min_alpha", 0.18)
        ),
        track_line_projection_enabled=bool(
            runtime_summary.get("track_line_projection_enabled", False)
        ),
        track_line_projection_min_points=int(
            runtime_summary.get("track_line_projection_min_points", 18)
        ),
        track_line_projection_history_frames=int(
            runtime_summary.get("track_line_projection_history_frames", 90)
        ),
        track_line_projection_blend=float(
            runtime_summary.get("track_line_projection_blend", 0.35)
        ),
        track_line_projection_max_shift_m=float(
            runtime_summary.get("track_line_projection_max_shift_m", 0.16)
        ),
        forward_smoothing_alpha=float(runtime_summary.get("track_forward_smoothing_alpha", 1.0)),
        forward_velocity_damping=float(runtime_summary.get("track_forward_velocity_damping", 1.0)),
        measurement_follow_enabled=bool(
            runtime_summary.get("track_measurement_follow_enabled", False)
        ),
        measurement_follow_blend=float(runtime_summary.get("track_measurement_follow_blend", 0.0)),
        measurement_follow_min_quality=float(
            runtime_summary.get("track_measurement_follow_min_quality", 0.0)
        ),
        measurement_follow_max_residual_m=float(
            runtime_summary.get("track_measurement_follow_max_residual_m", 0.0)
        ),
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
        motion_direction_gate_enabled=bool(
            runtime_summary.get("track_motion_direction_gate_enabled", False)
        ),
        motion_direction_min_speed_m_s=float(
            runtime_summary.get("track_motion_direction_min_speed_m_s", 0.18)
        ),
        motion_direction_min_displacement_m=float(
            runtime_summary.get("track_motion_direction_min_displacement_m", 0.35)
        ),
        motion_direction_max_angle_deg=float(
            runtime_summary.get("track_motion_direction_max_angle_deg", 105.0)
        ),
        motion_direction_max_cross_m=float(
            runtime_summary.get("track_motion_direction_max_cross_m", 0.75)
        ),
        motion_direction_cross_range_scale=float(
            runtime_summary.get("track_motion_direction_cross_range_scale", 0.04)
        ),
        max_object_count=int(_tracking_value(runtime_summary, "max_object_count", 3) or 0),
        expected_object_count=_tracking_value(runtime_summary, "expected_object_count", None),
        crossing_hold_frames=int(_tracking_value(runtime_summary, "crossing_hold_frames", 8) or 0),
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
        "cfg_path_source": cfg_path_source,
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
        "lateral_axis_sign_source": lateral_axis_sign_source,
        "angle_phase_sign_source": angle_phase_sign_source,
        "tdm_compensation_source": tdm_compensation_source,
        "tdm_phase_sign_source": tdm_phase_sign_source,
        "ablation_mode": ablation_mode,
        "tracker_enabled": tracker_enabled,
    }


def stage_cache_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "lab_data" / "stage_cache"


def _variant_cache_suffix(variant_label: str | None) -> str:
    if not variant_label:
        return ""
    label = str(variant_label).strip().lower()
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:80]


def _stage_cache_key(
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> str:
    mode = _normalize_ablation_mode(ablation_mode)
    base = str(session_id) if mode == "baseline" else f"{session_id}__{mode}"
    suffix = _variant_cache_suffix(variant_label)
    return f"{base}__{suffix}" if suffix else base


def stage_cache_dir(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> Path:
    return stage_cache_root(project_root) / _stage_cache_key(session_id, ablation_mode, variant_label)


def stage_cache_paths(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> dict[str, Path]:
    cache_dir = stage_cache_dir(project_root, session_id, ablation_mode, variant_label)
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


def load_stage_cache_manifest(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> dict | None:
    manifest_path = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)["manifest_path"]
    if not manifest_path.exists():
        return None
    return _load_json(manifest_path)


def load_stage_cache_frames(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> list[dict]:
    frames_path = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)["frames_path"]
    return _load_jsonl(frames_path)


def load_stage_features(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> list[dict]:
    features_path = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)["features_path"]
    return _load_jsonl(features_path)


def load_stage_feature_summary(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> dict | None:
    summary_path = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)["feature_summary_path"]
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def load_stage_traces(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> list[dict]:
    trace_path = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)["trace_path"]
    return _load_jsonl(trace_path)


def load_stage_trace_summary(
    project_root: Path,
    session_id: str,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> dict | None:
    summary_path = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)["trace_summary_path"]
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def load_stage_cache_frame(
    project_root: Path,
    session_id: str,
    ordinal: int,
    ablation_mode: str | None = "baseline",
    variant_label: str | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    cache_dir = stage_cache_dir(project_root, session_id, ablation_mode, variant_label)
    frames = load_stage_cache_frames(project_root, session_id, ablation_mode, variant_label)
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
    lead_step = feature.get("lead_step_m")
    lead_step_value = None if lead_step is None else float(lead_step)
    if bool(feature.get("lead_switch")):
        if lead_step_value is not None and lead_step_value >= 0.45:
            return "lead_id_switch", 8.5, f"lead id 변경과 함께 step={lead_step_value:.3f}m 점프가 발생했습니다."
        return "lead_id_switch", 7.0, "lead track id가 이전 프레임 대비 변경되었습니다."
    if lead_step_value is not None and lead_step_value >= 0.45:
        return "path_jump", 8.0, f"final lead step={lead_step_value:.3f}m로 큽니다."
    candidate_count = int(feature.get("detection_count") or 0)
    confirmed_count = int(feature.get("confirmed_track_count") or 0)
    if confirmed_count > 0 and candidate_count / max(confirmed_count, 1) >= 2.0:
        return "detection_over_split", 6.0, "confirmed track 대비 detection 후보가 많습니다."
    residual = feature.get("lead_measurement_residual_m")
    if residual is not None and float(residual) >= 1.0:
        return "representative_point_jump", 6.5, f"lead residual={float(residual):.3f}m로 매우 큽니다."
    if (
        residual is not None
        and float(residual) >= 0.45
        and lead_step_value is not None
        and lead_step_value >= 0.25
    ):
        return "representative_point_jump", 6.0, f"lead residual={float(residual):.3f}m, step={lead_step_value:.3f}m입니다."
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
                "rda_dense_points": int(_nested_get(detection, "rda_dense_points", "candidate_count", default=0) or 0),
                "cfar_projected_seeds": int(_nested_get(detection, "cfar", "projected_seed_count", default=0) or 0),
                "angle_passed": int(_nested_get(detection, "angle_validation", "passed_count", default=0) or 0),
                "coarse_merge_after": int(_nested_get(detection, "candidate_merge_coarse", "after_count", default=0) or 0),
                "body_refined": int(_nested_get(detection, "body_center_refinement", "refined_count", default=0) or 0),
                "blob_group_points": sum(
                    int((group.get("summary") or {}).get("component_point_count") or (group.get("summary") or {}).get("point_count") or 0)
                    for group in (_nested_get(detection, "blob_center_refinement", "groups", default=[]) or [])
                    if isinstance(group, dict)
                ),
                "blob_centered": int(_nested_get(detection, "blob_center_refinement", "output_count", default=0) or 0),
                "final_merge_after": int(_nested_get(detection, "candidate_merge_final", "after_count", default=0) or 0),
                "dbscan_output": int(_nested_get(detection, "dbscan", "output_count", default=0) or 0),
                "tracker_input": int(_nested_get(trace, "tracker_input_filter", "tracker_input_count", default=0) or 0),
                "association_matched": int(_nested_get(tracker, "association", "matched_count", default=0) or 0),
                "births": len(_nested_get(tracker, "track_lifecycle", "births", default=[]) or []),
                "deleted": len(_nested_get(tracker, "track_lifecycle", "deleted_track_ids", default=[]) or []),
                "display_confirmed": int(_nested_get(trace, "display_output", "confirmed_count", default=0) or 0),
                "rai_collapse_suspicious": int(
                    _nested_get(trace, "rai_collapse_diagnostics", "suspicious_count", default=0) or 0
                ),
            }
        )

    metrics = {}
    for key in [
        "cfar_candidates",
        "rda_dense_points",
        "cfar_projected_seeds",
        "angle_passed",
        "coarse_merge_after",
        "body_refined",
        "blob_group_points",
        "blob_centered",
        "final_merge_after",
        "dbscan_output",
        "tracker_input",
        "association_matched",
        "births",
        "deleted",
        "display_confirmed",
        "rai_collapse_suspicious",
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
    ablation_mode: str = "baseline",
    variant_label: str | None = None,
    cfg_path_override: str | Path | None = None,
    angle_phase_sign_override: float | None = None,
    lateral_axis_sign_override: float | None = None,
    tdm_compensation_override: bool | None = None,
    tdm_phase_sign_override: float | None = None,
) -> dict:
    project_root = Path(project_root).resolve()
    ablation_mode = _normalize_ablation_mode(ablation_mode)
    run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        registry.refresh_registry(project_root)
        run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        raise FileNotFoundError(f"Run session not found in registry: {session_id}")

    capture_dir = _resolve_capture_dir(project_root, run_detail)
    capture_manifest, _, _ = load_raw_capture(capture_dir)
    runtime_summary = _merge_runtime_summary(run_detail, capture_manifest)
    components = _build_runtime_components(
        project_root,
        runtime_summary,
        ablation_mode=ablation_mode,
        cfg_path_override=cfg_path_override,
        angle_phase_sign_override=angle_phase_sign_override,
        lateral_axis_sign_override=lateral_axis_sign_override,
        tdm_compensation_override=tdm_compensation_override,
        tdm_phase_sign_override=tdm_phase_sign_override,
    )

    paths = stage_cache_paths(project_root, session_id, ablation_mode, variant_label)
    requested_limit = int(frame_limit) if frame_limit not in (None, 0) else None
    if not force and paths["manifest_path"].exists() and paths["frames_path"].exists():
        existing_manifest = load_stage_cache_manifest(project_root, session_id, ablation_mode, variant_label) or {}
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
            tracker_enabled=components["tracker_enabled"],
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
        "cache_key": _stage_cache_key(session_id, ablation_mode, variant_label),
        "ablation_mode": ablation_mode,
        "variant_label": _variant_cache_suffix(variant_label) or None,
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
            "rai_collapse_diagnostics",
            "detection.rda_dense_points",
            "detection.cfar",
            "detection.cfar.projected_seeds",
            "detection.angle_validation",
            "detection.body_center_refinement",
            "detection.blob_center_refinement",
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
            "cfg_path_source": components["cfg_path_source"],
            "remove_static": bool(components["runtime_config"].remove_static),
            "doppler_guard_bins": int(components["runtime_config"].doppler_guard_bins),
            "range_resolution_m": round(float(components["runtime_config"].range_resolution_m), 6),
            "max_range_m": round(float(components["runtime_config"].max_range_m), 4),
            "range_fft_size": int(components["runtime_config"].range_fft_size),
            "doppler_fft_size": int(components["runtime_config"].doppler_fft_size),
            "angle_fft_size": int(components["runtime_config"].angle_fft_size),
            "lateral_axis_sign": float(components["runtime_config"].lateral_axis_sign),
            "lateral_axis_sign_source": components["lateral_axis_sign_source"],
            "angle_projection": str(getattr(components["runtime_config"], "angle_projection", "fft1d")),
            "angle_phase_sign": float(getattr(components["runtime_config"], "angle_phase_sign", -1.0)),
            "angle_phase_sign_source": components["angle_phase_sign_source"],
            "angle_source": str(
                getattr(
                    components["runtime_config"],
                    "angle_source",
                    components["detection_params"].get("angle_source", "collapsed_rai"),
                )
            ),
            "channel_calibration_enabled": bool(
                getattr(components["runtime_config"], "channel_calibration_enabled", False)
            ),
            "channel_calibration_count": int(
                len(getattr(components["runtime_config"], "channel_calibration_coefficients", ()) or ())
            ),
            "tdm_mimo_doppler_compensation_enabled": bool(
                getattr(components["runtime_config"], "tdm_mimo_doppler_compensation_enabled", False)
            ),
            "tdm_mimo_doppler_compensation_source": components["tdm_compensation_source"],
            "tdm_mimo_doppler_compensation_phase_sign": float(
                getattr(
                    components["runtime_config"],
                    "tdm_mimo_doppler_compensation_phase_sign",
                    1.0,
                )
            ),
            "tdm_mimo_doppler_compensation_phase_sign_source": components[
                "tdm_phase_sign_source"
            ],
            "tdm_mimo_doppler_compensation_slot_time_model": str(
                getattr(
                    components["runtime_config"],
                    "tdm_mimo_doppler_compensation_slot_time_model",
                    "uniform_tx_slot",
                )
            ),
            "tdm_mimo_doppler_compensation_reference_tx_slot": int(
                getattr(
                    components["runtime_config"],
                    "tdm_mimo_doppler_compensation_reference_tx_slot",
                    0,
                )
            ),
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
    parser.add_argument(
        "--variant-label",
        default=None,
        help="Optional cache suffix for diagnostic comparisons so variants do not overwrite each other.",
    )
    parser.add_argument(
        "--cfg-path-override",
        default=None,
        help="Optional cfg path override for replay diagnostics.",
    )
    parser.add_argument(
        "--angle-phase-sign",
        type=float,
        default=None,
        help="Optional angle_phase_sign override for replay diagnostics.",
    )
    parser.add_argument(
        "--lateral-axis-sign",
        type=float,
        default=None,
        choices=[-1.0, 1.0],
        help="Optional lateral axis sign override. Use -1 for invert_lateral_axis=true and 1 for false.",
    )
    parser.add_argument(
        "--tdm-compensation",
        default="auto",
        choices=["auto", "on", "off"],
        help="Override TDM-MIMO Doppler phase compensation for replay diagnostics.",
    )
    parser.add_argument(
        "--tdm-phase-sign",
        type=float,
        default=None,
        help="Optional TDM-MIMO Doppler compensation phase sign override.",
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=[
            "baseline",
            "doppler_slice_angle",
            "rda_candidates",
            "person_blob",
            "blob_center",
            "person_aware_merge",
            "multi_tracker_relaxed",
            "no_body_center",
            "no_duplicate_suppression",
            "no_merge",
            "no_dbscan",
            "tracker_off",
        ],
        help="Optional ablation mode for Stage Debug replay.",
    )
    args = parser.parse_args()

    manifest = build_stage_cache(
        Path(args.project_root),
        args.session,
        frame_limit=(args.limit or None),
        force=bool(args.force),
        ablation_mode=args.mode,
        variant_label=args.variant_label,
        cfg_path_override=args.cfg_path_override,
        angle_phase_sign_override=args.angle_phase_sign,
        lateral_axis_sign_override=args.lateral_axis_sign,
        tdm_compensation_override=(
            None if args.tdm_compensation == "auto" else args.tdm_compensation == "on"
        ),
        tdm_phase_sign_override=args.tdm_phase_sign,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
