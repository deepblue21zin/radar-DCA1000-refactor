from dataclasses import dataclass, field, replace
from datetime import datetime
import inspect
import json
from pathlib import Path
from queue import Empty, Full, Queue
import socket
import threading as th
import time

import numpy as np

from . import DSP
from .detection import detect_targets as _runtime_detect_targets
from .radar_runtime import (
    collapse_motion_rai,
    frame_to_radar_cube,
    integrate_rdi_channels,
    remove_static_clutter,
)
from .tracking import MultiTargetTracker
from .virtual_array import (
    apply_tdm_mimo_doppler_phase_compensation,
    cached_iwr6843isk_virtual_array,
    geometry_range_angle_from_fft as _runtime_geometry_range_angle_from_fft,
)


DCA1000_HEADER_BYTES = 10


def _filter_kwargs_for_loaded_signature(callable_obj, kwargs):
    parameters = inspect.signature(callable_obj).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def detect_targets(*args, **kwargs):
    return _runtime_detect_targets(*args, **_filter_kwargs_for_loaded_signature(_runtime_detect_targets, kwargs))


def geometry_range_angle_from_fft(*args, **kwargs):
    return _runtime_geometry_range_angle_from_fft(
        *args,
        **_filter_kwargs_for_loaded_signature(_runtime_geometry_range_angle_from_fft, kwargs),
    )


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    capture_ts: float
    assembled_ts: float
    iq: np.ndarray
    packets_in_frame: int
    sequence_start: int | None = None
    sequence_end: int | None = None
    byte_count_start: int | None = None
    byte_count_end: int | None = None
    udp_gap_count: int = 0
    byte_mismatch_count: int = 0
    out_of_sequence_count: int = 0
    invalid: bool = False
    invalid_reason: str = ""
    processed_ts: float | None = None
    rdi: np.ndarray | None = None
    rai: np.ndarray | None = None
    detections: tuple = field(default_factory=tuple)
    tracker_input_count: int = 0
    track_birth_blocked: bool = False
    tracker_policy: str = "full"
    confirmed_tracks: tuple = field(default_factory=tuple)
    tentative_tracks: tuple = field(default_factory=tuple)
    stage_timings_ms: dict = field(default_factory=dict)


def _serialize_detection(detection):
    return {
        "range_bin": int(detection.range_bin),
        "doppler_bin": int(detection.doppler_bin),
        "angle_bin": int(detection.angle_bin),
        "range_m": round(float(detection.range_m), 4),
        "angle_deg": round(float(detection.angle_deg), 3),
        "x_m": round(float(detection.x_m), 4),
        "y_m": round(float(detection.y_m), 4),
        "rdi_peak": round(float(detection.rdi_peak), 4),
        "rai_peak": round(float(detection.rai_peak), 4),
        "score": round(float(detection.score), 4),
    }


def _serialize_track(track):
    display_id = getattr(track, "display_id", None)
    return {
        "track_id": int(track.track_id),
        "display_id": int(display_id if display_id is not None else track.track_id),
        "is_primary": bool(track.is_primary),
        "doppler_bin": int(track.doppler_bin),
        "range_m": round(float(track.range_m), 4),
        "angle_deg": round(float(track.angle_deg), 3),
        "x_m": round(float(track.x_m), 4),
        "y_m": round(float(track.y_m), 4),
        "vx_m_s": round(float(track.vx_m_s), 4),
        "vy_m_s": round(float(track.vy_m_s), 4),
        "rdi_peak": round(float(track.rdi_peak), 4),
        "rai_peak": round(float(track.rai_peak), 4),
        "score": round(float(track.score), 4),
        "confidence": round(float(track.confidence), 4),
        "age": int(track.age),
        "hits": int(track.hits),
        "misses": int(track.misses),
        "measurement_quality": round(float(track.measurement_quality), 4),
        "measurement_residual_m": round(float(track.measurement_residual_m), 4),
    }


def _append_jsonl(log_path, record):
    if log_path is None:
        return
    log_path = Path(log_path)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl_records(log_path):
    records = []
    log_path = Path(log_path)
    if not log_path.exists():
        return records
    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _round_stage_timings(stage_timings_ms):
    return {
        key: round(float(value), 3)
        for key, value in (stage_timings_ms or {}).items()
        if value is not None
    }


def _apply_xy_coordinate_correction(x_m, y_m, runtime_config):
    yaw_deg = float(getattr(runtime_config, "xy_yaw_correction_deg", 0.0) or 0.0)
    lateral_offset_m = float(getattr(runtime_config, "xy_lateral_offset_m", 0.0) or 0.0)
    forward_offset_m = float(getattr(runtime_config, "xy_forward_offset_m", 0.0) or 0.0)
    if abs(yaw_deg) <= 1e-9 and abs(lateral_offset_m) <= 1e-9 and abs(forward_offset_m) <= 1e-9:
        return float(x_m), float(y_m)

    yaw_rad = float(np.radians(yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    x_value = float(x_m)
    y_value = float(y_m)
    corrected_x = (x_value * cos_yaw) - (y_value * sin_yaw) + lateral_offset_m
    corrected_y = (x_value * sin_yaw) + (y_value * cos_yaw) + forward_offset_m
    return float(corrected_x), float(corrected_y)


def _correct_detection_coordinates(detection, runtime_config):
    corrected_x, corrected_y = _apply_xy_coordinate_correction(
        detection.x_m,
        detection.y_m,
        runtime_config,
    )
    if corrected_x == float(detection.x_m) and corrected_y == float(detection.y_m):
        return detection
    corrected_range_m = float(np.hypot(corrected_x, corrected_y))
    corrected_angle_deg = float(np.degrees(np.arctan2(corrected_x, max(corrected_y, 1e-6))))
    return replace(
        detection,
        range_m=corrected_range_m,
        angle_deg=corrected_angle_deg,
        x_m=corrected_x,
        y_m=corrected_y,
    )


def _correct_detection_sequence(detections, runtime_config):
    yaw_deg = float(getattr(runtime_config, "xy_yaw_correction_deg", 0.0) or 0.0)
    lateral_offset_m = float(getattr(runtime_config, "xy_lateral_offset_m", 0.0) or 0.0)
    forward_offset_m = float(getattr(runtime_config, "xy_forward_offset_m", 0.0) or 0.0)
    if abs(yaw_deg) <= 1e-9 and abs(lateral_offset_m) <= 1e-9 and abs(forward_offset_m) <= 1e-9:
        return list(detections)
    return [_correct_detection_coordinates(item, runtime_config) for item in detections]


def select_tracker_input_for_frame(
    frame_packet,
    detections,
    *,
    block_track_birth_on_invalid=True,
    invalid_policy=None,
):
    policy_name = "full"
    allow_track_birth = True
    tracker_detections = list(detections)
    if not frame_packet.invalid:
        return tracker_detections, allow_track_birth, policy_name

    invalid_policy = invalid_policy or {}
    drop_gap_threshold = int(invalid_policy.get("drop_gap_threshold", 0))
    drop_seq_threshold = int(invalid_policy.get("drop_out_of_sequence_threshold", 0))
    drop_byte_threshold = int(invalid_policy.get("drop_byte_mismatch_threshold", 0))
    birth_block_gap_threshold = int(invalid_policy.get("birth_block_gap_threshold", 0))
    birth_block_seq_threshold = int(invalid_policy.get("birth_block_out_of_sequence_threshold", 0))
    birth_block_byte_threshold = int(invalid_policy.get("birth_block_byte_mismatch_threshold", 0))

    severe_invalid = (
        frame_packet.udp_gap_count >= drop_gap_threshold > 0
        or frame_packet.out_of_sequence_count >= drop_seq_threshold > 0
        or frame_packet.byte_mismatch_count >= drop_byte_threshold > 0
    )
    moderate_invalid = (
        frame_packet.udp_gap_count >= birth_block_gap_threshold > 0
        or frame_packet.out_of_sequence_count >= birth_block_seq_threshold > 0
        or frame_packet.byte_mismatch_count >= birth_block_byte_threshold > 0
    )

    if severe_invalid:
        tracker_detections = []
        allow_track_birth = False
        policy_name = "drop"
    elif moderate_invalid and block_track_birth_on_invalid:
        allow_track_birth = False
        policy_name = "no_birth"

    return tracker_detections, allow_track_birth, policy_name


def project_range_angle_cube(range_doppler_fft, runtime_config):
    projection = str(getattr(runtime_config, "angle_projection", "fft1d") or "fft1d").strip().lower()
    angle_input_fft = range_doppler_fft
    if bool(getattr(runtime_config, "tdm_mimo_doppler_compensation_enabled", False)):
        angle_input_fft = apply_tdm_mimo_doppler_phase_compensation(
            range_doppler_fft,
            tx_num=int(getattr(runtime_config, "tx_num", 1)),
            phase_sign=float(
                getattr(runtime_config, "tdm_mimo_doppler_compensation_phase_sign", 1.0)
            ),
            reference_tx_slot=int(
                getattr(runtime_config, "tdm_mimo_doppler_compensation_reference_tx_slot", 0)
            ),
            slot_time_model=str(
                getattr(
                    runtime_config,
                    "tdm_mimo_doppler_compensation_slot_time_model",
                    "uniform_tx_slot",
                )
            ),
        )
    if projection in ("iwr6843isk_geometry_2d", "isk_geometry_2d", "geometry_2d"):
        model = cached_iwr6843isk_virtual_array(
            str(runtime_config.config_path),
            int(runtime_config.rx_num),
            int(runtime_config.tx_num),
        )
        rai_cube = geometry_range_angle_from_fft(
            angle_input_fft,
            raw_order=model.raw_order,
            x_lambda=model.x_lambda,
            z_lambda=model.z_lambda,
            azimuth_axis_rad=runtime_config.angle_axis_rad,
            elevation_axis_deg=runtime_config.angle_elevation_axis_deg,
            phase_sign=runtime_config.angle_phase_sign,
            channel_coefficients=(
                getattr(runtime_config, "channel_calibration_coefficients", ())
                if getattr(runtime_config, "channel_calibration_enabled", False)
                else None
            ),
        )
        return np.fft.fftshift(rai_cube, axes=0)

    if projection not in ("fft1d", "angle_fft", "legacy_fft"):
        raise ValueError(f"Unsupported angle_projection: {projection}")

    rai_cube = DSP.range_angle_from_fft(
        angle_input_fft,
        mode=1,
        angle_fft_size=runtime_config.angle_fft_size,
    )
    return np.fft.fftshift(rai_cube, axes=0)


def _sign_label(value, deadband=0.05):
    if value is None:
        return "none"
    value = float(value)
    if value > deadband:
        return "right"
    if value < -deadband:
        return "left"
    return "center"


def _wrap_angle_delta_deg(left, right):
    if left is None or right is None:
        return None
    return float((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _angle_roi_mask_for_range(runtime_config, range_m, detection_region):
    angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
    x_axis = float(range_m) * np.sin(angle_axis)
    y_axis = float(range_m) * np.cos(angle_axis)
    return (
        (np.abs(x_axis) <= float(detection_region.lateral_limit_m))
        & (y_axis >= float(detection_region.min_forward_m))
        & (y_axis <= float(detection_region.forward_limit_m))
    )


def _peak_from_angle_profile(profile, runtime_config, range_m, angle_mask):
    values = np.asarray(profile, dtype=np.float64)
    if values.size == 0:
        return None
    mask = np.asarray(angle_mask, dtype=bool)
    if mask.shape != values.shape or not np.any(mask):
        mask = np.ones(values.shape, dtype=bool)
    masked = np.where(mask, values, 0.0)
    peak_bin = int(np.argmax(masked))
    peak_power = float(masked[peak_bin])
    angle_rad = float(np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)[peak_bin])
    x_m = float(range_m) * float(np.sin(angle_rad))
    y_m = float(range_m) * float(np.cos(angle_rad))
    return {
        "angle_bin": peak_bin,
        "angle_deg": float(np.degrees(angle_rad)),
        "power": peak_power,
        "x_m": x_m,
        "y_m": y_m,
        "side": _sign_label(x_m),
    }


def _round_optional(value, digits=4):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def _build_rai_collapse_diagnostics(
    *,
    detection_trace,
    rai_cube,
    rai_map,
    runtime_config,
    detection_region,
    max_rows=12,
    angle_delta_threshold_deg=15.0,
    x_delta_threshold_m=0.35,
):
    cfar_candidates = (
        ((detection_trace or {}).get("cfar") or {}).get("top_candidates") or []
    )
    rows = []
    for rank, candidate in enumerate(cfar_candidates[: int(max_rows)], start=1):
        try:
            range_bin = int(candidate.get("range_bin"))
            doppler_bin = int(candidate.get("doppler_bin"))
        except (TypeError, ValueError):
            continue
        if range_bin < 0 or range_bin >= np.asarray(rai_map).shape[0]:
            continue
        if np.asarray(rai_cube).ndim != 3 or doppler_bin < 0 or doppler_bin >= np.asarray(rai_cube).shape[0]:
            continue
        range_m = float(candidate.get("range_m", runtime_config.range_axis_m[range_bin]))
        angle_mask = _angle_roi_mask_for_range(runtime_config, range_m, detection_region)
        slice_peak = _peak_from_angle_profile(
            np.asarray(rai_cube)[doppler_bin, range_bin, :],
            runtime_config,
            range_m,
            angle_mask,
        )
        collapsed_peak = _peak_from_angle_profile(
            np.asarray(rai_map)[range_bin, :],
            runtime_config,
            range_m,
            angle_mask,
        )
        if not slice_peak or not collapsed_peak:
            continue
        angle_delta = _wrap_angle_delta_deg(collapsed_peak["angle_deg"], slice_peak["angle_deg"])
        x_delta = float(collapsed_peak["x_m"] - slice_peak["x_m"])
        sign_changed = slice_peak["side"] != "center" and collapsed_peak["side"] not in {
            "center",
            slice_peak["side"],
        }
        suspicious = (
            abs(float(angle_delta)) >= float(angle_delta_threshold_deg)
            or abs(float(x_delta)) >= float(x_delta_threshold_m)
            or bool(sign_changed)
        )
        rows.append(
            {
                "candidate_rank": int(rank),
                "range_bin": range_bin,
                "range_m": _round_optional(range_m),
                "doppler_bin": doppler_bin,
                "rdi_power": _round_optional(candidate.get("power")),
                "slice_angle_deg": _round_optional(slice_peak["angle_deg"], 3),
                "slice_x_m": _round_optional(slice_peak["x_m"]),
                "slice_side": slice_peak["side"],
                "collapsed_angle_deg": _round_optional(collapsed_peak["angle_deg"], 3),
                "collapsed_x_m": _round_optional(collapsed_peak["x_m"]),
                "collapsed_side": collapsed_peak["side"],
                "angle_delta_deg": _round_optional(angle_delta, 3),
                "x_delta_m": _round_optional(x_delta),
                "x_sign_changed": bool(sign_changed),
                "suspicious": bool(suspicious),
            }
        )
    suspicious_count = sum(1 for row in rows if row.get("suspicious"))
    return {
        "rows": rows,
        "row_count": int(len(rows)),
        "suspicious_count": int(suspicious_count),
        "angle_delta_threshold_deg": float(angle_delta_threshold_deg),
        "x_delta_threshold_m": float(x_delta_threshold_m),
    }


def process_frame_packet(
    raw_frame,
    *,
    runtime_config,
    detection_region,
    min_range_bin,
    max_range_bin,
    tracker: MultiTargetTracker,
    block_track_birth_on_invalid=True,
    invalid_policy=None,
    detection_params=None,
    capture_stage_timing=True,
    return_artifacts=False,
    capture_trace=False,
    tracker_enabled=True,
):
    loop_started = time.perf_counter()
    stage_timings_ms = {}
    frame_trace = None
    if capture_trace:
        frame_trace = {
            "trace_version": 1,
            "frame_id": int(raw_frame.frame_id),
            "raw_udp_packets": {
                "packets_in_frame": int(raw_frame.packets_in_frame),
                "sequence_start": raw_frame.sequence_start,
                "sequence_end": raw_frame.sequence_end,
                "byte_count_start": raw_frame.byte_count_start,
                "byte_count_end": raw_frame.byte_count_end,
            },
            "frame_parsing": {
                "invalid": bool(raw_frame.invalid),
                "invalid_reason": raw_frame.invalid_reason,
                "udp_gap_count": int(raw_frame.udp_gap_count),
                "byte_mismatch_count": int(raw_frame.byte_mismatch_count),
                "out_of_sequence_count": int(raw_frame.out_of_sequence_count),
                "iq_sample_count": int(np.asarray(raw_frame.iq).size),
            },
        }

    stage_started = time.perf_counter()
    radar_cube = frame_to_radar_cube(raw_frame.iq, runtime_config)
    stage_timings_ms["cube_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["radar_cube"] = {
            "shape": [int(dim) for dim in np.asarray(radar_cube).shape],
            "mean_abs": round(float(np.mean(np.abs(radar_cube))), 4),
            "max_abs": round(float(np.max(np.abs(radar_cube))), 4),
        }
    if runtime_config.remove_static:
        stage_started = time.perf_counter()
        radar_cube = remove_static_clutter(radar_cube)
        stage_timings_ms["static_ms"] = (time.perf_counter() - stage_started) * 1000.0
        if frame_trace is not None:
            frame_trace["static_removal"] = {
                "enabled": True,
                "output_mean_abs": round(float(np.mean(np.abs(radar_cube))), 4),
                "output_max_abs": round(float(np.max(np.abs(radar_cube))), 4),
            }
    else:
        stage_timings_ms["static_ms"] = 0.0
        if frame_trace is not None:
            frame_trace["static_removal"] = {"enabled": False}

    stage_started = time.perf_counter()
    shared_range_doppler_fft = DSP.shared_range_doppler_fft(
        radar_cube,
        padding_size=[
            runtime_config.doppler_fft_size,
            runtime_config.range_fft_size,
        ],
    )
    stage_timings_ms["shared_fft2_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["shared_fft"] = {
            "shape": [int(dim) for dim in np.asarray(shared_range_doppler_fft).shape],
            "mean_abs": round(float(np.mean(np.abs(shared_range_doppler_fft))), 4),
            "max_abs": round(float(np.max(np.abs(shared_range_doppler_fft))), 4),
        }

    stage_started = time.perf_counter()
    rdi_cube = DSP.range_doppler_from_fft(
        shared_range_doppler_fft,
        mode=1,
    )
    stage_timings_ms["range_doppler_project_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["rdi_cube"] = {
            "shape": [int(dim) for dim in np.asarray(rdi_cube).shape],
        }

    stage_started = time.perf_counter()
    rai_cube = project_range_angle_cube(
        shared_range_doppler_fft,
        runtime_config,
    )
    stage_timings_ms["range_angle_project_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["rai_cube"] = {
            "angle_projection": getattr(runtime_config, "angle_projection", "fft1d"),
            "doppler_axis": "fftshifted",
            "angle_source": getattr(runtime_config, "angle_source", "collapsed_rai"),
            "angle_bias_correction_enabled": bool(
                getattr(runtime_config, "angle_bias_correction_enabled", False)
            ),
            "angle_bias_correction_mode": str(
                getattr(runtime_config, "angle_bias_correction_mode", "toward_center")
            ),
            "angle_bias_left_deg": round(
                float(getattr(runtime_config, "angle_bias_left_deg", 0.0)),
                4,
            ),
            "angle_bias_center_deg": round(
                float(getattr(runtime_config, "angle_bias_center_deg", 0.0)),
                4,
            ),
            "angle_bias_right_deg": round(
                float(getattr(runtime_config, "angle_bias_right_deg", 0.0)),
                4,
            ),
            "angle_bias_center_band_deg": round(
                float(getattr(runtime_config, "angle_bias_center_band_deg", 7.0)),
                4,
            ),
            "line_deskew_correction_enabled": bool(
                getattr(runtime_config, "line_deskew_correction_enabled", False)
            ),
            "line_deskew_correction_diagnostic_only": bool(
                getattr(runtime_config, "line_deskew_correction_diagnostic_only", True)
            ),
            "line_deskew_gain": round(
                float(getattr(runtime_config, "line_deskew_gain", 0.3)),
                4,
            ),
            "line_deskew_max_shift_m": round(
                float(getattr(runtime_config, "line_deskew_max_shift_m", 0.08)),
                4,
            ),
            "line_deskew_min_history_frames": int(
                getattr(runtime_config, "line_deskew_min_history_frames", 20)
            ),
            "range_angle_correction_enabled": bool(
                getattr(runtime_config, "range_angle_correction_enabled", False)
            ),
            "range_angle_correction_diagnostic_only": bool(
                getattr(runtime_config, "range_angle_correction_diagnostic_only", True)
            ),
            "range_angle_reference_half_width_m": round(
                float(
                    getattr(
                        runtime_config,
                        "range_angle_correction_reference_half_width_m",
                        3.5,
                    )
                ),
                4,
            ),
            "range_angle_reference_forward_m": round(
                float(
                    getattr(
                        runtime_config,
                        "range_angle_correction_reference_forward_m",
                        7.0,
                    )
                ),
                4,
            ),
            "range_angle_correction_max_delta_deg": round(
                float(getattr(runtime_config, "range_angle_correction_max_delta_deg", 6.0)),
                4,
            ),
            "channel_calibration_enabled": bool(
                getattr(runtime_config, "channel_calibration_enabled", False)
            ),
            "tdm_mimo_doppler_compensation_enabled": bool(
                getattr(runtime_config, "tdm_mimo_doppler_compensation_enabled", False)
            ),
            "tdm_mimo_doppler_compensation_phase_sign": round(
                float(getattr(runtime_config, "tdm_mimo_doppler_compensation_phase_sign", 1.0)),
                3,
            ),
            "tdm_mimo_doppler_compensation_slot_time_model": str(
                getattr(
                    runtime_config,
                    "tdm_mimo_doppler_compensation_slot_time_model",
                    "uniform_tx_slot",
                )
            ),
            "tdm_mimo_doppler_compensation_reference_tx_slot": int(
                getattr(runtime_config, "tdm_mimo_doppler_compensation_reference_tx_slot", 0)
            ),
            "xy_yaw_correction_deg": round(
                float(getattr(runtime_config, "xy_yaw_correction_deg", 0.0) or 0.0),
                4,
            ),
            "xy_lateral_offset_m": round(
                float(getattr(runtime_config, "xy_lateral_offset_m", 0.0) or 0.0),
                4,
            ),
            "xy_forward_offset_m": round(
                float(getattr(runtime_config, "xy_forward_offset_m", 0.0) or 0.0),
                4,
            ),
            "shape": [int(dim) for dim in np.asarray(rai_cube).shape],
        }

    stage_started = time.perf_counter()
    rdi = integrate_rdi_channels(rdi_cube)
    stage_timings_ms["integrate_rdi_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["rdi"] = {
            "shape": [int(dim) for dim in np.asarray(rdi).shape],
            "max": round(float(np.max(rdi)), 4),
            "mean": round(float(np.mean(rdi)), 4),
        }

    stage_started = time.perf_counter()
    rai = collapse_motion_rai(
        rai_cube,
        guard_bins=runtime_config.doppler_guard_bins,
    )
    stage_timings_ms["collapse_rai_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["rai"] = {
            "shape": [int(dim) for dim in np.asarray(rai).shape],
            "max": round(float(np.max(rai)), 4),
            "mean": round(float(np.mean(rai)), 4),
        }

    stage_started = time.perf_counter()
    detection_trace = {} if frame_trace is not None else None
    detections = detect_targets(
        rdi,
        rai,
        runtime_config,
        min_range_bin,
        max_range_bin,
        detection_region,
        rai_cube=rai_cube,
        trace=detection_trace,
        **dict(detection_params or {}),
    )
    detections = _correct_detection_sequence(detections, runtime_config)
    stage_timings_ms["detect_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["coordinate_correction"] = {
            "xy_yaw_correction_deg": round(
                float(getattr(runtime_config, "xy_yaw_correction_deg", 0.0) or 0.0),
                4,
            ),
            "xy_lateral_offset_m": round(
                float(getattr(runtime_config, "xy_lateral_offset_m", 0.0) or 0.0),
                4,
            ),
            "xy_forward_offset_m": round(
                float(getattr(runtime_config, "xy_forward_offset_m", 0.0) or 0.0),
                4,
            ),
            "corrected_detection_count": len(detections),
        }
        frame_trace["detection"] = detection_trace or {}
        frame_trace["rai_collapse_diagnostics"] = _build_rai_collapse_diagnostics(
            detection_trace=detection_trace or {},
            rai_cube=rai_cube,
            rai_map=rai,
            runtime_config=runtime_config,
            detection_region=detection_region,
        )

    stage_started = time.perf_counter()
    tracker_detections, allow_track_birth, tracker_policy = select_tracker_input_for_frame(
        raw_frame,
        detections,
        block_track_birth_on_invalid=block_track_birth_on_invalid,
        invalid_policy=invalid_policy,
    )
    stage_timings_ms["tracker_gate_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["tracker_input_filter"] = {
            "policy": tracker_policy,
            "input_detection_count": len(detections),
            "tracker_input_count": len(tracker_detections),
            "allow_track_birth": bool(allow_track_birth),
            "track_birth_blocked": bool(not allow_track_birth),
        }

    stage_started = time.perf_counter()
    tracker_trace = {} if frame_trace is not None else None
    if tracker_enabled:
        confirmed_tracks, tentative_tracks = tracker.update(
            tracker_detections,
            frame_ts=raw_frame.capture_ts,
            allow_track_birth=allow_track_birth,
            rai_map=rai,
            runtime_config=runtime_config,
            trace=tracker_trace,
        )
    else:
        confirmed_tracks, tentative_tracks = [], []
        if tracker_trace is not None:
            tracker_trace.update(
                {
                    "trace_version": 1,
                    "disabled": True,
                    "input_detection_count": len(detections),
                    "measurement_count": len(tracker_detections),
                    "measurements": [_serialize_detection(item) for item in tracker_detections[:12]],
                }
            )
    stage_timings_ms["track_ms"] = (time.perf_counter() - stage_started) * 1000.0
    if frame_trace is not None:
        frame_trace["tracker"] = tracker_trace or {}
    processed_ts = time.perf_counter()
    stage_timings_ms["compute_total_ms"] = (processed_ts - loop_started) * 1000.0
    if frame_trace is not None:
        frame_trace["display_output"] = {
            "confirmed_count": len(confirmed_tracks),
            "tentative_count": len(tentative_tracks),
            "confirmed_track_ids": [int(track.track_id) for track in confirmed_tracks],
            "tentative_track_ids": [int(track.track_id) for track in tentative_tracks],
            "confirmed_tracks": [_serialize_track(track) for track in confirmed_tracks[:12]],
            "tentative_tracks": [_serialize_track(track) for track in tentative_tracks[:12]],
        }
        frame_trace["stage_timings_ms"] = _round_stage_timings(stage_timings_ms)

    processed_frame = replace(
        raw_frame,
        processed_ts=processed_ts,
        rdi=rdi,
        rai=rai,
        detections=tuple(detections),
        tracker_input_count=len(tracker_detections),
        track_birth_blocked=not allow_track_birth,
        tracker_policy=tracker_policy,
        confirmed_tracks=tuple(confirmed_tracks),
        tentative_tracks=tuple(tentative_tracks),
        stage_timings_ms=_round_stage_timings(stage_timings_ms) if capture_stage_timing else {},
    )

    if not return_artifacts:
        return processed_frame, None

    artifacts = {
        "radar_cube_shape": tuple(int(dim) for dim in radar_cube.shape),
        "shared_fft_shape": tuple(int(dim) for dim in shared_range_doppler_fft.shape),
        "rdi_cube_shape": tuple(int(dim) for dim in np.asarray(rdi_cube).shape),
        "rai_cube_shape": tuple(int(dim) for dim in np.asarray(rai_cube).shape),
        "cube_preview": np.asarray(np.abs(radar_cube).mean(axis=2), dtype=np.float32),
        "rdi": np.asarray(rdi, dtype=np.float32),
        "rai": np.asarray(rai, dtype=np.float32),
        "tracker_input_detections": tuple(tracker_detections),
        "frame_trace": frame_trace,
    }
    return processed_frame, artifacts


def _put_latest(queue_object, item):
    while True:
        try:
            queue_object.put_nowait(item)
            return
        except Full:
            try:
                queue_object.get_nowait()
            except Empty:
                return


def _parse_dca1000_packet(packet_bytes):
    if len(packet_bytes) < DCA1000_HEADER_BYTES:
        return None, None, b""

    sequence_id = int.from_bytes(packet_bytes[:4], byteorder="little", signed=False)
    byte_count = int.from_bytes(packet_bytes[4:10], byteorder="little", signed=False)
    return sequence_id, byte_count, packet_bytes[DCA1000_HEADER_BYTES:]


class RawFrameCaptureWriter:
    def __init__(self, raw_data_path, raw_index_path, *, max_queue_frames=256):
        self.raw_data_path = Path(raw_data_path)
        self.raw_index_path = Path(raw_index_path)
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_data_file = self.raw_data_path.open("ab")
        self.raw_index_file = self.raw_index_path.open("a", encoding="utf-8", buffering=1)
        self.first_capture_ts = None
        self.bytes_written = int(self.raw_data_path.stat().st_size if self.raw_data_path.exists() else 0)
        self._queue = Queue(maxsize=max(int(max_queue_frames), 1))
        self._closed = False
        self._backpressure_logged = False
        self._worker = th.Thread(
            target=self._writer_loop,
            name="RawCaptureWriter",
            daemon=True,
        )
        self._worker.start()

    def write_frame(self, frame_packet: FramePacket):
        if self._closed:
            return
        try:
            self._queue.put_nowait(frame_packet)
        except Full:
            if not self._backpressure_logged:
                print(
                    "Warning: raw capture writer queue is full; "
                    "blocking UDP listener until disk catches up."
                )
                self._backpressure_logged = True
            self._queue.put(frame_packet)

    def _writer_loop(self):
        while True:
            frame_packet = self._queue.get()
            try:
                if frame_packet is None:
                    return
                self._write_frame_sync(frame_packet)
            finally:
                self._queue.task_done()

    def _write_frame_sync(self, frame_packet: FramePacket):
        iq_payload = np.asarray(frame_packet.iq, dtype=np.dtype("<i2")).tobytes(order="C")
        byte_offset = self.bytes_written
        self.raw_data_file.write(iq_payload)
        self.bytes_written += len(iq_payload)

        if self.first_capture_ts is None:
            self.first_capture_ts = float(frame_packet.capture_ts)

        record = {
            "frame_id": int(frame_packet.frame_id),
            "capture_ts": round(float(frame_packet.capture_ts), 6),
            "assembled_ts": round(float(frame_packet.assembled_ts), 6),
            "capture_elapsed_s": round(float(frame_packet.capture_ts - self.first_capture_ts), 6),
            "assembled_elapsed_s": round(float(frame_packet.assembled_ts - self.first_capture_ts), 6),
            "byte_offset": int(byte_offset),
            "byte_length": int(len(iq_payload)),
            "sample_count": int(frame_packet.iq.size),
            "packets_in_frame": int(frame_packet.packets_in_frame),
            "sequence_start": frame_packet.sequence_start,
            "sequence_end": frame_packet.sequence_end,
            "byte_count_start": frame_packet.byte_count_start,
            "byte_count_end": frame_packet.byte_count_end,
            "udp_gap_count": int(frame_packet.udp_gap_count),
            "byte_mismatch_count": int(frame_packet.byte_mismatch_count),
            "out_of_sequence_count": int(frame_packet.out_of_sequence_count),
            "invalid": bool(frame_packet.invalid),
            "invalid_reason": frame_packet.invalid_reason,
        }
        self.raw_index_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            print("Warning: raw capture writer did not finish within 10s.")
        if self.raw_data_file is not None:
            self.raw_data_file.close()
            self.raw_data_file = None
        if self.raw_index_file is not None:
            self.raw_index_file.close()
            self.raw_index_file = None


def load_raw_capture(capture_dir):
    capture_dir = Path(capture_dir)
    capture_manifest_path = capture_dir / "capture_manifest.json"
    raw_index_path = capture_dir / "raw_frames_index.jsonl"
    raw_data_path = capture_dir / "raw_frames.i16"
    if not capture_dir.exists():
        raise FileNotFoundError(f"Replay capture directory not found: {capture_dir}")
    if not capture_manifest_path.exists():
        raise FileNotFoundError(f"Replay capture manifest not found: {capture_manifest_path}")
    if not raw_index_path.exists():
        raise FileNotFoundError(f"Replay capture index not found: {raw_index_path}")
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Replay capture data not found: {raw_data_path}")

    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    index_records = _load_jsonl_records(raw_index_path)
    if not index_records:
        raise RuntimeError(f"No replay frames found in {raw_index_path}")
    return capture_manifest, index_records, raw_data_path


def iter_raw_capture_frame_packets(capture_dir):
    capture_manifest, index_records, raw_data_path = load_raw_capture(capture_dir)
    frame_length_hint = int((capture_manifest.get("raw_capture") or {}).get("frame_length_samples") or 0)
    with raw_data_path.open("rb") as raw_data_file:
        for record in index_records:
            raw_data_file.seek(int(record["byte_offset"]))
            payload = raw_data_file.read(int(record["byte_length"]))
            iq = np.frombuffer(payload, dtype=np.dtype("<i2")).astype(np.int16, copy=True)
            if frame_length_hint > 0 and iq.size != frame_length_hint:
                print(
                    "Warning: replay frame length mismatch "
                    f"(frame_id={record.get('frame_id')}, expected={frame_length_hint}, actual={iq.size})"
                )

            capture_ts = float(record.get("capture_ts", 0.0))
            assembled_ts = float(record.get("assembled_ts", capture_ts))
            yield FramePacket(
                frame_id=int(record.get("frame_id", 0)),
                capture_ts=capture_ts,
                assembled_ts=assembled_ts,
                iq=iq,
                packets_in_frame=int(record.get("packets_in_frame", 1)),
                sequence_start=record.get("sequence_start"),
                sequence_end=record.get("sequence_end"),
                byte_count_start=record.get("byte_count_start"),
                byte_count_end=record.get("byte_count_end"),
                udp_gap_count=int(record.get("udp_gap_count", 0)),
                byte_mismatch_count=int(record.get("byte_mismatch_count", 0)),
                out_of_sequence_count=int(record.get("out_of_sequence_count", 0)),
                invalid=bool(record.get("invalid", False)),
                invalid_reason=str(record.get("invalid_reason", "")),
            )


class RawCaptureReplaySource(th.Thread):
    def __init__(self, name, frame_queue, capture_dir, playback_speed=1.0, loop=False, autostart=False):
        th.Thread.__init__(self, name=name)
        self.frame_queue = frame_queue
        self.capture_dir = Path(capture_dir)
        self.playback_speed = max(float(playback_speed), 0.01)
        self.loop = bool(loop)
        self._stop_requested = th.Event()
        self._start_requested = th.Event()
        if autostart:
            self._start_requested.set()
        self.capture_manifest_path = self.capture_dir / "capture_manifest.json"
        self.raw_index_path = self.capture_dir / "raw_frames_index.jsonl"
        self.raw_data_path = self.capture_dir / "raw_frames.i16"

    def start_streaming(self):
        self._start_requested.set()

    def close(self):
        self._stop_requested.set()
        self._start_requested.set()

    def _load_capture(self):
        capture_manifest, index_records, _ = load_raw_capture(self.capture_dir)
        return capture_manifest, index_records

    def run(self):
        capture_manifest, index_records = self._load_capture()
        frame_length_hint = int((capture_manifest.get("raw_capture") or {}).get("frame_length_samples") or 0)

        while not self._stop_requested.is_set():
            if not self._start_requested.wait(timeout=0.1):
                continue

            replay_started_at = time.perf_counter()
            with self.raw_data_path.open("rb") as raw_data_file:
                for record in index_records:
                    if self._stop_requested.is_set():
                        break

                    capture_elapsed_s = float(record.get("capture_elapsed_s", 0.0))
                    assembled_elapsed_s = float(record.get("assembled_elapsed_s", capture_elapsed_s))
                    scheduled_capture_ts = replay_started_at + (capture_elapsed_s / self.playback_speed)
                    while not self._stop_requested.is_set():
                        remaining_s = scheduled_capture_ts - time.perf_counter()
                        if remaining_s <= 0:
                            break
                        time.sleep(min(remaining_s, 0.01))

                    raw_data_file.seek(int(record["byte_offset"]))
                    payload = raw_data_file.read(int(record["byte_length"]))
                    iq = np.frombuffer(payload, dtype=np.dtype("<i2")).astype(np.int16, copy=True)
                    if frame_length_hint > 0 and iq.size != frame_length_hint:
                        print(
                            "Warning: replay frame length mismatch "
                            f"(frame_id={record.get('frame_id')}, expected={frame_length_hint}, actual={iq.size})"
                        )

                    assembled_delta_s = max(assembled_elapsed_s - capture_elapsed_s, 0.0)
                    frame_packet = FramePacket(
                        frame_id=int(record.get("frame_id", 0)),
                        capture_ts=scheduled_capture_ts,
                        assembled_ts=scheduled_capture_ts + (assembled_delta_s / self.playback_speed),
                        iq=iq,
                        packets_in_frame=int(record.get("packets_in_frame", 1)),
                        sequence_start=record.get("sequence_start"),
                        sequence_end=record.get("sequence_end"),
                        byte_count_start=record.get("byte_count_start"),
                        byte_count_end=record.get("byte_count_end"),
                        udp_gap_count=int(record.get("udp_gap_count", 0)),
                        byte_mismatch_count=int(record.get("byte_mismatch_count", 0)),
                        out_of_sequence_count=int(record.get("out_of_sequence_count", 0)),
                        invalid=bool(record.get("invalid", False)),
                        invalid_reason=str(record.get("invalid_reason", "")),
                    )
                    _put_latest(self.frame_queue, frame_packet)

            if not self.loop:
                break

        _put_latest(self.frame_queue, None)


class UdpListener(th.Thread):
    def __init__(self, name, frame_queue, data_frame_length, data_address, buff_size, raw_capture_writer=None):
        """
        :param name: str
                        Object name

        :param frame_queue: queue object
                        A queue used to store assembled frame packets

        :param data_frame_length: int
                        Length of a single frame in int16 samples

        :param data_address: (str, int)
                        Address for binding udp stream, str for host IP address, int for host data port

        :param buff_size: int
                        Requested OS UDP receive buffer size
        """
        th.Thread.__init__(self, name=name)
        self.frame_queue = frame_queue
        self.frame_length = data_frame_length
        self.data_address = data_address
        self.socket_buffer_size = max(int(buff_size), 65536)
        self.raw_capture_writer = raw_capture_writer
        self._stop_requested = th.Event()
        self._data_socket = None
        # DCA1000 packets are small; keep the per-read buffer modest and use
        # SO_RCVBUF to absorb bursts at the OS socket layer.
        self.recv_packet_size = 4096

    def close(self):
        self._stop_requested.set()
        if self._data_socket is not None:
            try:
                self._data_socket.close()
            except OSError:
                pass
            self._data_socket = None
        if self.raw_capture_writer is not None:
            self.raw_capture_writer.close()
            self.raw_capture_writer = None

    def run(self):
        dt = np.dtype(np.int16).newbyteorder("<")
        sample_buffer = []
        frame_count = 0
        first_packet_logged = False

        current_capture_ts = None
        current_packet_count = 0
        current_gap_count = 0
        current_byte_mismatch_count = 0
        current_out_of_sequence_count = 0
        current_invalid = False
        current_invalid_reasons = []
        current_sequence_start = None
        current_byte_count_start = None

        last_sequence = None
        last_byte_count = None
        last_payload_size = None

        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._data_socket = data_socket
        try:
            try:
                data_socket.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_RCVBUF,
                    self.socket_buffer_size,
                )
            except OSError as exc:
                print(f"Warning: failed to apply SO_RCVBUF={self.socket_buffer_size}: {exc}")
            data_socket.bind(self.data_address)
            data_socket.settimeout(0.5)
            print("Create socket successfully")
            actual_socket_buffer = data_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(
                "UDP socket receive buffer "
                f"(requested={self.socket_buffer_size}, actual={actual_socket_buffer})"
            )
            print("Now start data streaming")

            while not self._stop_requested.is_set():
                try:
                    packet_bytes, addr = data_socket.recvfrom(self.recv_packet_size)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_requested.is_set():
                        break
                    raise

                recv_ts = time.perf_counter()
                if not first_packet_logged:
                    print(f"Received first UDP packet from {addr}")
                    first_packet_logged = True

                sequence_id, byte_count, payload = _parse_dca1000_packet(packet_bytes)
                if sequence_id is None:
                    continue

                if current_capture_ts is None:
                    current_capture_ts = recv_ts
                    current_sequence_start = sequence_id
                    current_byte_count_start = byte_count

                packet_gap_count = 0
                packet_byte_mismatch_count = 0
                packet_out_of_sequence_count = 0
                packet_invalid = False
                packet_invalid_reasons = []

                if last_sequence is not None:
                    sequence_delta = sequence_id - last_sequence
                    if sequence_delta != 1:
                        packet_out_of_sequence_count = 1
                        if sequence_delta > 1:
                            packet_gap_count = sequence_delta - 1
                        packet_invalid = True
                        packet_invalid_reasons.append("sequence")

                    if last_byte_count is not None and last_payload_size is not None:
                        expected_byte_count = last_byte_count + last_payload_size
                        if byte_count != expected_byte_count:
                            packet_byte_mismatch_count = 1
                            packet_invalid = True
                            packet_invalid_reasons.append("byte_count")

                if len(payload) % dt.itemsize != 0:
                    payload = payload[: len(payload) - (len(payload) % dt.itemsize)]
                    packet_invalid = True
                    packet_invalid_reasons.append("payload_alignment")

                if payload:
                    sample_buffer.extend(np.frombuffer(payload, dtype=dt))

                current_packet_count += 1
                current_gap_count += packet_gap_count
                current_byte_mismatch_count += packet_byte_mismatch_count
                current_out_of_sequence_count += packet_out_of_sequence_count
                current_invalid = current_invalid or packet_invalid
                for reason in packet_invalid_reasons:
                    if reason not in current_invalid_reasons:
                        current_invalid_reasons.append(reason)

                last_sequence = sequence_id
                last_byte_count = byte_count
                last_payload_size = len(payload)

                while len(sample_buffer) >= self.frame_length:
                    frame_count += 1
                    if frame_count == 1:
                        print("Received first complete radar frame")

                    frame_packet = FramePacket(
                        frame_id=frame_count,
                        capture_ts=current_capture_ts if current_capture_ts is not None else recv_ts,
                        assembled_ts=recv_ts,
                        iq=np.asarray(sample_buffer[: self.frame_length], dtype=np.int16),
                        packets_in_frame=max(current_packet_count, 1),
                        sequence_start=current_sequence_start,
                        sequence_end=sequence_id,
                        byte_count_start=current_byte_count_start,
                        byte_count_end=byte_count,
                        udp_gap_count=current_gap_count,
                        byte_mismatch_count=current_byte_mismatch_count,
                        out_of_sequence_count=current_out_of_sequence_count,
                        invalid=current_invalid,
                        invalid_reason=",".join(current_invalid_reasons),
                    )
                    if self.raw_capture_writer is not None:
                        self.raw_capture_writer.write_frame(frame_packet)
                    _put_latest(self.frame_queue, frame_packet)
                    sample_buffer = sample_buffer[self.frame_length :]

                    if sample_buffer:
                        current_capture_ts = recv_ts
                        current_packet_count = 1
                        current_gap_count = packet_gap_count
                        current_byte_mismatch_count = packet_byte_mismatch_count
                        current_out_of_sequence_count = packet_out_of_sequence_count
                        current_invalid = packet_invalid
                        current_invalid_reasons = list(packet_invalid_reasons)
                        current_sequence_start = sequence_id
                        current_byte_count_start = byte_count
                    else:
                        current_capture_ts = None
                        current_packet_count = 0
                        current_gap_count = 0
                        current_byte_mismatch_count = 0
                        current_out_of_sequence_count = 0
                        current_invalid = False
                        current_invalid_reasons = []
                        current_sequence_start = None
                        current_byte_count_start = None
        finally:
            self._data_socket = None
            if self.raw_capture_writer is not None:
                self.raw_capture_writer.close()
                self.raw_capture_writer = None


class DataProcessor(th.Thread):
    def __init__(
        self,
        name,
        config,
        raw_frame_queue,
        processed_frame_queue,
        detection_region,
        min_range_bin,
        max_range_bin,
        tracker: MultiTargetTracker,
        block_track_birth_on_invalid=True,
        invalid_policy=None,
        processed_frame_log_path=None,
        detection_params=None,
        write_processed_frames=True,
        include_payloads=True,
        capture_stage_timing=True,
        tracker_enabled=True,
    ):
        """
        :param name: str
                        Object name

        :param config: RadarRuntimeConfig
                        Parsed radar runtime config

        :param raw_frame_queue: queue object
                        A queue for access data received by UdpListener

        :param processed_frame_queue: queue object
                        A queue for processed frame packets
        """
        th.Thread.__init__(self, name=name)
        self.runtime_config = config
        self.raw_frame_queue = raw_frame_queue
        self.processed_frame_queue = processed_frame_queue
        self.detection_region = detection_region
        self.min_range_bin = min_range_bin
        self.max_range_bin = max_range_bin
        self.tracker = tracker
        self.block_track_birth_on_invalid = block_track_birth_on_invalid
        self.invalid_policy = invalid_policy or {}
        self.processed_frame_log_path = (
            Path(processed_frame_log_path)
            if processed_frame_log_path is not None
            else None
        )
        self.detection_params = dict(detection_params or {})
        self.write_processed_frames = bool(write_processed_frames)
        self.include_payloads = bool(include_payloads)
        self.capture_stage_timing = bool(capture_stage_timing)
        self.tracker_enabled = bool(tracker_enabled)

    def close(self):
        _put_latest(self.raw_frame_queue, None)

    def select_tracker_input(self, frame_packet, detections):
        return select_tracker_input_for_frame(
            frame_packet,
            detections,
            block_track_birth_on_invalid=self.block_track_birth_on_invalid,
            invalid_policy=self.invalid_policy,
        )

    def build_processed_record(self, frame_packet):
        capture_to_process_ms = None
        if frame_packet.processed_ts is not None:
            capture_to_process_ms = max(
                (frame_packet.processed_ts - frame_packet.capture_ts) * 1000.0,
                0.0,
            )

        record = {
            "frame_id": int(frame_packet.frame_id),
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "capture_ts": round(frame_packet.capture_ts, 6),
            "assembled_ts": round(frame_packet.assembled_ts, 6),
            "processed_ts": None
            if frame_packet.processed_ts is None
            else round(frame_packet.processed_ts, 6),
            "capture_to_process_ms": None
            if capture_to_process_ms is None
            else round(capture_to_process_ms, 3),
            "packets_in_frame": int(frame_packet.packets_in_frame),
            "sequence_start": frame_packet.sequence_start,
            "sequence_end": frame_packet.sequence_end,
            "byte_count_start": frame_packet.byte_count_start,
            "byte_count_end": frame_packet.byte_count_end,
            "udp_gap_count": int(frame_packet.udp_gap_count),
            "byte_mismatch_count": int(frame_packet.byte_mismatch_count),
            "out_of_sequence_count": int(frame_packet.out_of_sequence_count),
            "invalid": bool(frame_packet.invalid),
            "invalid_reason": frame_packet.invalid_reason,
            "track_birth_blocked": bool(frame_packet.track_birth_blocked),
            "tracker_policy": frame_packet.tracker_policy,
            "candidate_count": len(frame_packet.detections),
            "tracker_input_count": int(frame_packet.tracker_input_count),
            "confirmed_track_count": len(frame_packet.confirmed_tracks),
            "tentative_track_count": len(frame_packet.tentative_tracks),
        }
        if frame_packet.stage_timings_ms:
            record["stage_timings_ms"] = _round_stage_timings(frame_packet.stage_timings_ms)
        if self.include_payloads:
            record["detections"] = [
                _serialize_detection(detection)
                for detection in frame_packet.detections
            ]
            record["confirmed_tracks"] = [
                _serialize_track(track)
                for track in frame_packet.confirmed_tracks
            ]
            record["tentative_tracks"] = [
                _serialize_track(track)
                for track in frame_packet.tentative_tracks
            ]
        return record

    def log_processed_frame(self, frame_packet):
        if not self.write_processed_frames:
            return
        record = self.build_processed_record(frame_packet)
        _append_jsonl(self.processed_frame_log_path, record)

    def run(self):
        frame_count = 0
        while True:
            raw_frame = self.raw_frame_queue.get()
            if raw_frame is None:
                break
            frame_count += 1
            loop_started = time.perf_counter()
            processed_frame, _ = process_frame_packet(
                raw_frame,
                runtime_config=self.runtime_config,
                detection_region=self.detection_region,
                min_range_bin=self.min_range_bin,
                max_range_bin=self.max_range_bin,
                tracker=self.tracker,
                block_track_birth_on_invalid=self.block_track_birth_on_invalid,
                invalid_policy=self.invalid_policy,
                detection_params=self.detection_params,
                capture_stage_timing=self.capture_stage_timing,
                return_artifacts=False,
                tracker_enabled=self.tracker_enabled,
            )

            if frame_count == 1:
                print("Generated first processed RDI/RAI frame")
                print(f"Initial detection candidates: {len(processed_frame.detections)}")

            log_write_ms = 0.0
            if self.write_processed_frames:
                log_started = time.perf_counter()
                self.log_processed_frame(processed_frame)
                log_write_ms = (time.perf_counter() - log_started) * 1000.0

            if self.capture_stage_timing:
                updated_stage_timings = dict(processed_frame.stage_timings_ms)
                updated_stage_timings["log_write_ms"] = round(log_write_ms, 3)
                updated_stage_timings["pipeline_total_ms"] = round(
                    (time.perf_counter() - loop_started) * 1000.0,
                    3,
                )
                processed_frame = replace(
                    processed_frame,
                    stage_timings_ms=updated_stage_timings,
                )

            _put_latest(self.processed_frame_queue, processed_frame)
