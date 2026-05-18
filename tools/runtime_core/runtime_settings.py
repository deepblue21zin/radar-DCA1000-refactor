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
    "radar_board": "IWR6843ISK",
    "config_path": "config/profile_isk_3d_100ms_txorder.cfg",
    "tuning_path": "config/live_motion_tuning_isk.json",
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
        "angle_projection": "fft1d",
        "angle_elevation_min_deg": -40.0,
        "angle_elevation_max_deg": 40.0,
        "angle_elevation_step_deg": 4.0,
        "angle_phase_sign": -1.0,
        "angle_source": "collapsed_rai",
        "coordinate_correction": {
            "yaw_deg": 0.0,
            "lateral_offset_m": 0.0,
            "forward_offset_m": 0.0,
        },
        "angle_bias_correction": {
            "enabled": False,
            "mode": "toward_center",
            "left_deg": 0.0,
            "center_deg": 0.0,
            "right_deg": 0.0,
            "center_band_deg": 7.0,
        },
        "line_deskew_correction": {
            "enabled": False,
            "diagnostic_only": True,
            "gain": 0.3,
            "max_shift_m": 0.08,
            "min_history_frames": 20,
            "max_history_frames": 90,
            "min_y_span_m": 0.35,
        },
        "range_angle_correction": {
            "enabled": False,
            "diagnostic_only": True,
            "reference_half_width_m": 3.5,
            "reference_forward_m": 7.0,
            "range_bins_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "angle_bins_norm": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "delta_table_deg": [],
            "max_delta_deg": 6.0,
        },
        "channel_calibration": {
            "enabled": False,
            "coefficients": [],
        },
        "tdm_mimo_doppler_compensation": {
            "enabled": False,
            "phase_sign": 1.0,
            "slot_time_model": "uniform_tx_slot",
            "reference_tx_slot": 0,
        },
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
            "enable_body_center_refinement": True,
            "enable_candidate_merge": True,
            "enable_dbscan": True,
            "duplicate_suppression_enabled": True,
            "duplicate_suppression_radius_m": 0.55,
            "duplicate_suppression_range_scale": 0.03,
            "duplicate_suppression_doppler_bins": 6,
            "duplicate_suppression_score_ratio": 0.82,
            "range_doppler_ambiguity_suppression_enabled": False,
            "range_doppler_ambiguity_range_tolerance_m": 0.30,
            "range_doppler_ambiguity_doppler_bins": 2,
            "range_doppler_ambiguity_min_angle_delta_deg": 16.0,
            "range_doppler_ambiguity_min_separation_m": 0.75,
            "range_doppler_ambiguity_mirror_x_tolerance_m": 0.45,
            "range_doppler_ambiguity_mirror_y_tolerance_m": 0.45,
            "range_doppler_ambiguity_min_abs_angle_deg": 4.0,
            "rd_cfar_output_guard_enabled": False,
            "rd_cfar_output_guard_max_references": 12,
            "rd_cfar_output_guard_min_reference_score_ratio": 0.05,
            "rd_cfar_output_guard_range_tolerance_m": 0.45,
            "rd_cfar_output_guard_doppler_bins": 3,
            "rd_cfar_output_guard_prefer_references": False,
            "rd_cfar_output_guard_replace_shifted": True,
            "rd_cfar_output_guard_replace_shift_m": 0.45,
            "rd_cfar_output_guard_fallback_to_references": True,
            "object_count_estimator_enabled": True,
            "object_count_max_objects": 3,
            "object_count_min_separation_m": 0.65,
            "object_count_min_doppler_bins": 7,
            "object_count_min_score_ratio": 0.05,
            "protect_multi_object_candidates": False,
            "limit_output_to_object_count": False,
            "min_output_score": 0.0,
            "person_blob_refinement_enabled": False,
            "person_blob_doppler_radius_bins": 2,
            "person_blob_min_points": 4,
            "person_blob_floor_quantile": 0.65,
            "person_blob_center_method": "weighted_median",
            "person_blob_peak_blend": 0.10,
            "blob_center_refinement_enabled": False,
            "blob_center_max_candidates": 36,
            "blob_center_min_points": 2,
            "blob_center_min_score_ratio": 0.04,
            "blob_center_cluster_radius_m": 0.65,
            "blob_center_cluster_radius_range_scale": 0.04,
            "blob_center_cluster_radius_bands": [
                {"r_min": 0.0, "r_max": 1.5, "radius_m": 0.55},
                {"r_min": 1.5, "r_max": 3.0, "radius_m": 0.72},
                {"r_min": 3.0, "r_max": None, "radius_m": 0.90},
            ],
            "blob_center_doppler_radius_bins": 10,
            "blob_center_method": "weighted_median_trimmed",
            "blob_center_trim_radius_m": 0.85,
            "blob_center_floor_quantile": 0.65,
            "blob_center_peak_blend": 0.0,
            "blob_center_single_min_score_ratio": 0.12,
            "blob_center_single_range_window_m": 1.05,
            "blob_center_single_side_deadband_m": 0.15,
            "blob_center_cube_range_radius_m": None,
            "blob_center_cube_angle_radius_deg": None,
            "blob_center_cube_relative_floor": None,
            "blob_center_dense_enabled": True,
            "blob_center_dense_quantile": 0.995,
            "blob_center_dense_min_normalized_power": 0.08,
            "blob_center_dense_max_points": 2400,
            "blob_center_dense_min_points": 6,
            "blob_center_dense_grouping_mode": "rd_primary",
            "blob_center_dense_angle_radius_deg": 18.0,
            "blob_center_dense_angle_floor_quantile": 0.70,
            "blob_center_dense_angle_relative_floor": 0.25,
            "blob_center_anchor_max_shift_m": 0.45,
            "blob_center_anchor_blend": 0.75,
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
        "lateral_measurement_scale": 1.0,
        "forward_measurement_scale": 1.0,
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
        "lateral_range_damping_enabled": False,
        "lateral_range_damping_start_m": 1.4,
        "lateral_range_damping_full_m": 3.8,
        "lateral_range_damping_min_alpha": 0.18,
        "line_projection_enabled": False,
        "line_projection_min_points": 18,
        "line_projection_history_frames": 90,
        "line_projection_blend": 0.35,
        "line_projection_max_shift_m": 0.16,
        "forward_smoothing_alpha": 1.0,
        "forward_velocity_damping": 1.0,
        "motion_correction_strength": 1.0,
        "measurement_follow_enabled": False,
        "measurement_follow_blend": 0.0,
        "measurement_follow_min_quality": 0.0,
        "measurement_follow_max_residual_m": 0.0,
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
        "motion_direction_gate_enabled": False,
        "motion_direction_min_speed_m_s": 0.18,
        "motion_direction_min_displacement_m": 0.35,
        "motion_direction_max_angle_deg": 105.0,
        "motion_direction_max_cross_m": 0.75,
        "motion_direction_cross_range_scale": 0.04,
        "max_object_count": 3,
        "expected_object_count": None,
        "crossing_hold_frames": 8,
        "output_smoothing_enabled": False,
        "output_smoothing_alpha": 0.35,
        "output_smoothing_max_step_m": 0.18,
        "output_smoothing_reset_m": 1.2,
        "output_smoothing_min_hits": 3,
        "recent_lost_track_memory_frames": 0,
        "reactivation_gate_m": 0.0,
        "reactivation_direction_weight": 0.0,
        "reactivation_doppler_gate_bins": 0,
        "display_id_stitching_enabled": False,
        "display_id_stitching_gate_m": 0.75,
        "display_id_stitching_memory_frames": 30,
        "display_id_stitching_direction_weight": 0.25,
        "display_id_stitching_doppler_gate_bins": 0,
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
        "tentative_requires_multi_evidence": True,
        "tentative_multi_evidence_min_tracker_inputs": 2,
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
    "radar_board",
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
    if runtime_settings_path is None:
        runtime_settings_path = os.environ.get("RADAR_RUNTIME_SETTINGS_PATH")
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
