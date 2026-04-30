import json
import os
from copy import deepcopy
from pathlib import Path


DEFAULT_STATIC_SETTINGS = {
    "cli_baudrate": 115200,
    "network": {
        "host_ip": "192.168.33.30",
        "data_port": 4098,
        "config_port": 4096,
        "fpga_ip": "192.168.33.180",
        "fpga_port": 4096,
        "buffer_size": 2097152,
    },
    "dca": {
        "config_timeout_s": 2.0,
        "packet_size_bytes": 1472,
        "packet_delay_us": 100,
        "packet_delay_ticks_per_us": 125,
    },
    "spatial_view": {
        "height": 180,
        "y": 42,
        "point_base_z_m": 0.10,
        "point_confidence_scale_m": 1.10,
    },
}

DEFAULT_RUNTIME_SETTINGS = {
    "config_path": "config/profile_3d.cfg",
    "tuning_path": "config/live_motion_tuning.json",
    "cli_port": "COM11",
    "logging": {
        "enabled": True,
        "variant": "baseline",
        "scenario_id": "",
        "input_mode": "live",
        "source_capture": "",
        "notes": "",
        "capture_duration_s": None,
        "write_raw_capture": True,
        "raw_capture_root": "logs/raw",
        "write_processed_frames": True,
        "write_render_frames": True,
        "write_status_log": True,
        "write_event_log": True,
        "include_payloads": True,
        "capture_system_snapshot": True,
        "capture_stage_timing": True,
        "report_generation_mode": "deferred",
    },
}

DEFAULT_TUNING_SETTINGS = {
    "processing": {
        "remove_static": True,
        "doppler_guard_bins": 2,
        "invert_lateral_axis": False,
    },
    "roi": {
        "lateral_m": 1.5,
        "forward_m": 3.0,
        "min_forward_m": 0.25,
    },
    "detection": {
        "allow_strongest_fallback": False,
        "max_targets": 1,
        "display_min_confidence": 0.26,
        "cluster_min_samples": 2,
        "cluster_velocity_weight": 0.0,
        "algorithm": {
            "cfar_training_cells": [6, 6],
            "cfar_guard_cells": [1, 1],
            "cfar_scale": 5.0,
            "global_quantile": 0.985,
            "angle_quantile": 0.75,
            "angle_contrast_scale": 1.35,
            "min_cartesian_separation_m": 0.65,
            "angle_centroid_radius_bands": [
                {"r_min": 0.0, "r_max": 1.5, "radius": 1},
                {"r_min": 1.5, "r_max": 3.0, "radius": 2},
                {"r_min": 3.0, "r_max": None, "radius": 3},
            ],
            "body_center_patch_bands": [
                {
                    "r_min": 0.0,
                    "r_max": 1.5,
                    "range_radius_bins": 1,
                    "angle_radius_bins": 2,
                    "relative_floor": 0.60,
                },
                {
                    "r_min": 1.5,
                    "r_max": 3.0,
                    "range_radius_bins": 2,
                    "angle_radius_bins": 3,
                    "relative_floor": 0.55,
                },
                {
                    "r_min": 3.0,
                    "r_max": None,
                    "range_radius_bins": 3,
                    "angle_radius_bins": 4,
                    "relative_floor": 0.50,
                },
            ],
            "candidate_merge_bands": [
                {
                    "r_min": 0.0,
                    "r_max": 1.5,
                    "merge_radius_m": 0.32,
                    "range_bin_radius": 1,
                    "doppler_bin_radius": 2,
                },
                {
                    "r_min": 1.5,
                    "r_max": 3.0,
                    "merge_radius_m": 0.48,
                    "range_bin_radius": 2,
                    "doppler_bin_radius": 3,
                },
                {
                    "r_min": 3.0,
                    "r_max": None,
                    "merge_radius_m": 0.62,
                    "range_bin_radius": 3,
                    "doppler_bin_radius": 4,
                },
            ],
            "duplicate_suppression_enabled": True,
            "duplicate_suppression_radius_m": 0.55,
            "duplicate_suppression_range_scale": 0.03,
            "duplicate_suppression_doppler_bins": 6,
            "duplicate_suppression_score_ratio": 0.82,
        },
        "dbscan_adaptive_eps_bands": [
            {"r_min": 0.25, "r_max": 1.0, "eps": 0.34, "min_samples": 2},
            {"r_min": 1.0, "r_max": 2.0, "eps": 0.44, "min_samples": 2},
            {"r_min": 2.0, "r_max": 3.5, "eps": 0.56, "min_samples": 2},
        ],
    },
    "tracking": {
        "confirm_hits": 3,
        "max_misses": 4,
        "process_var": 1.0,
        "measurement_var": 0.43,
        "range_measurement_scale": 0.50,
        "confidence_measurement_scale": 0.35,
        "association_gate": 5.99,
        "doppler_zero_guard_bins": 3,
        "doppler_gate_bins": 18,
        "doppler_cost_weight": 0.65,
        "report_miss_tolerance": 1,
        "lost_gate_factor": 1.3,
        "tentative_gate_factor": 0.65,
        "birth_suppression_radius_m": 0.55,
        "primary_track_birth_scale": 1.35,
        "birth_suppression_weak_radius_scale": 1.0,
        "birth_suppression_score_ratio": 0.0,
        "birth_suppression_confidence_ratio": 0.0,
        "birth_suppression_doppler_bins": 0,
        "birth_suppression_miss_tolerance": 3,
        "primary_track_hold_frames": 4,
        "lateral_deadband_m": 0.05,
        "lateral_deadband_range_scale": 0.03,
        "lateral_smoothing_alpha": 0.45,
        "lateral_velocity_damping": 0.55,
        "local_remeasurement_enabled": True,
        "local_remeasurement_blend": 0.35,
        "local_remeasurement_max_shift_m": 0.28,
        "local_remeasurement_track_bias": 0.15,
        "local_remeasurement_patch_bands": [
            {
                "r_min": 0.0,
                "r_max": 1.5,
                "range_radius_bins": 1,
                "angle_radius_bins": 2,
                "relative_floor": 0.55,
            },
            {
                "r_min": 1.5,
                "r_max": 3.0,
                "range_radius_bins": 2,
                "angle_radius_bins": 3,
                "relative_floor": 0.50,
            },
            {
                "r_min": 3.0,
                "r_max": None,
                "range_radius_bins": 3,
                "angle_radius_bins": 4,
                "relative_floor": 0.48,
            },
        ],
        "measurement_soft_gate_enabled": True,
        "measurement_soft_gate_floor": 0.35,
        "measurement_soft_gate_start_m": 0.16,
        "measurement_soft_gate_full_m": 0.52,
        "measurement_soft_gate_range_scale": 0.05,
        "measurement_soft_gate_speed_scale": 0.06,
    },
    "pipeline": {
        "queue_size": 4,
        "block_track_birth_on_invalid": True,
        "invalid_policy": {
            "birth_block_gap_threshold": 16,
            "birth_block_out_of_sequence_threshold": 2,
            "birth_block_byte_mismatch_threshold": 2,
            "drop_gap_threshold": 140,
            "drop_out_of_sequence_threshold": 6,
            "drop_byte_mismatch_threshold": 6,
        },
    },
    "visualization": {
        "show_tentative_tracks": True,
        "tentative_min_confidence": 0.30,
        "tentative_min_hits": 2,
        "display_hysteresis_frames": 5,
        "display_hysteresis_confidence_floor": 0.12,
        "display_primary_bonus_frames": 3,
    },
}

STATIC_SECTION_KEYS = (
    "cli_baudrate",
    "network",
    "dca",
    "spatial_view",
)

RUNTIME_SECTION_KEYS = (
    "config_path",
    "tuning_path",
    "cli_port",
    "logging",
)

TUNING_SECTION_KEYS = (
    "processing",
    "roi",
    "detection",
    "tracking",
    "pipeline",
    "visualization",
)


def _deep_merge(base_value, override_value):
    if isinstance(base_value, dict) and isinstance(override_value, dict):
        merged = deepcopy(base_value)
        for key, value in override_value.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override_value)


def _load_json_if_exists(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_config_path(project_root, path_value, default_relative_path):
    if path_value is None:
        return Path(project_root) / default_relative_path

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def build_settings_snapshot(settings, section_keys):
    return {
        section_key: deepcopy(settings[section_key])
        for section_key in section_keys
        if section_key in settings
    }


def build_default_settings():
    settings = {}
    settings = _deep_merge(settings, DEFAULT_STATIC_SETTINGS)
    settings = _deep_merge(settings, DEFAULT_RUNTIME_SETTINGS)
    settings = _deep_merge(settings, DEFAULT_TUNING_SETTINGS)
    return settings


def load_runtime_settings(
    project_root,
    runtime_settings_path=None,
    tuning_path=None,
    static_settings_path=None,
    settings_path=None,
):
    if runtime_settings_path is None and settings_path is not None:
        runtime_settings_path = settings_path

    project_root = Path(project_root)
    resolved_static_settings_path = _resolve_config_path(
        project_root,
        static_settings_path,
        "config/live_motion_static_settings.json",
    )
    resolved_runtime_settings_path = _resolve_config_path(
        project_root,
        runtime_settings_path,
        "config/live_motion_runtime_settings.json",
    )

    settings = build_default_settings()
    settings = _deep_merge(settings, _load_json_if_exists(resolved_static_settings_path))
    settings = _deep_merge(settings, _load_json_if_exists(resolved_runtime_settings_path))

    env_tuning_path = os.environ.get("RADAR_TUNING_PATH")
    resolved_tuning_path = _resolve_config_path(
        project_root,
        tuning_path if tuning_path is not None else (env_tuning_path or settings.get("tuning_path")),
        "config/live_motion_tuning.json",
    )
    settings = _deep_merge(settings, _load_json_if_exists(resolved_tuning_path))

    settings["_static_settings_path"] = str(resolved_static_settings_path)
    settings["_runtime_settings_path"] = str(resolved_runtime_settings_path)
    settings["_settings_path"] = str(resolved_runtime_settings_path)
    settings["_tuning_path"] = str(resolved_tuning_path)
    settings["static"] = build_settings_snapshot(settings, STATIC_SECTION_KEYS)
    settings["runtime"] = build_settings_snapshot(settings, RUNTIME_SECTION_KEYS)
    settings["tuning"] = build_settings_snapshot(settings, TUNING_SECTION_KEYS)
    settings["_config_path_resolved"] = str(
        resolve_project_path(project_root, settings["config_path"])
    )
    return settings


def resolve_project_path(project_root, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(project_root) / path
