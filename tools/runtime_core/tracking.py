"""Lightweight multi-target Kalman tracker for radar cluster centroids."""

from math import atan2, cos, degrees, hypot, isfinite, radians
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detection_core.refinement import refine_body_center_from_patch as _refine_body_center_from_patch
from .tracking_core.assignment import linear_sum_assignment as _linear_sum_assignment
from .tracking_core.kalman import load_filterpy as _load_filterpy
from .tracking_core.types import Track as _Track, TrackEstimate, TrackState


def _nearest_axis_bin(axis_values, value) -> int:
    return int(np.argmin(np.abs(np.asarray(axis_values) - float(value))))


class MultiTargetTracker:
    def __init__(
        self,
        process_var=1.0,
        measurement_var=0.4,
        range_measurement_scale=0.0,
        confidence_measurement_scale=0.0,
        angle_resolution_rad=0.0,
        lateral_measurement_scale=1.0,
        forward_measurement_scale=1.0,
        association_gate=5.99,
        doppler_center_bin=None,
        doppler_zero_guard_bins=2,
        doppler_gate_bins=0,
        doppler_cost_weight=0.0,
        max_missed_frames=8,
        min_confirmed_hits=2,
        report_miss_tolerance=2,
        lost_gate_factor=1.2,
        tentative_gate_factor=0.5,
        birth_suppression_radius_m=0.0,
        primary_track_birth_scale=1.0,
        birth_suppression_weak_radius_scale=1.0,
        birth_suppression_score_ratio=0.0,
        birth_suppression_confidence_ratio=0.0,
        birth_suppression_doppler_bins=0,
        birth_suppression_miss_tolerance=0,
        primary_track_hold_frames=0,
        lateral_deadband_m=0.0,
        lateral_deadband_range_scale=0.0,
        lateral_smoothing_alpha=1.0,
        lateral_velocity_damping=1.0,
        lateral_range_damping_enabled=False,
        lateral_range_damping_start_m=1.4,
        lateral_range_damping_full_m=3.8,
        lateral_range_damping_min_alpha=0.18,
        track_line_projection_enabled=False,
        track_line_projection_min_points=18,
        track_line_projection_history_frames=90,
        track_line_projection_blend=0.35,
        track_line_projection_max_shift_m=0.16,
        forward_smoothing_alpha=1.0,
        forward_velocity_damping=1.0,
        measurement_follow_enabled=False,
        measurement_follow_blend=0.0,
        measurement_follow_min_quality=0.0,
        measurement_follow_max_residual_m=0.0,
        local_remeasurement_enabled=False,
        local_remeasurement_blend=0.0,
        local_remeasurement_max_shift_m=0.0,
        local_remeasurement_track_bias=0.0,
        local_remeasurement_patch_bands=None,
        measurement_soft_gate_enabled=True,
        measurement_soft_gate_floor=0.35,
        measurement_soft_gate_start_m=0.16,
        measurement_soft_gate_full_m=0.52,
        measurement_soft_gate_range_scale=0.05,
        measurement_soft_gate_speed_scale=0.06,
        motion_direction_gate_enabled=False,
        motion_direction_min_speed_m_s=0.18,
        motion_direction_min_displacement_m=0.35,
        motion_direction_max_angle_deg=105.0,
        motion_direction_max_cross_m=0.75,
        motion_direction_cross_range_scale=0.04,
        max_object_count=0,
        expected_object_count=None,
        crossing_hold_frames=0,
    ):
        if process_var <= 0:
            raise ValueError("process_var must be positive.")
        if measurement_var <= 0:
            raise ValueError("measurement_var must be positive.")
        if range_measurement_scale < 0:
            raise ValueError("range_measurement_scale must be non-negative.")
        if confidence_measurement_scale < 0:
            raise ValueError("confidence_measurement_scale must be non-negative.")
        if angle_resolution_rad < 0:
            raise ValueError("angle_resolution_rad must be non-negative.")
        if lateral_measurement_scale < 0:
            raise ValueError("lateral_measurement_scale must be non-negative.")
        if forward_measurement_scale <= 0:
            raise ValueError("forward_measurement_scale must be positive.")
        if association_gate <= 0:
            raise ValueError("association_gate must be positive.")
        if doppler_zero_guard_bins < 0:
            raise ValueError("doppler_zero_guard_bins must be non-negative.")
        if doppler_gate_bins < 0:
            raise ValueError("doppler_gate_bins must be non-negative.")
        if doppler_cost_weight < 0:
            raise ValueError("doppler_cost_weight must be non-negative.")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative.")
        if min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be at least 1.")
        if report_miss_tolerance < 0:
            raise ValueError("report_miss_tolerance must be non-negative.")
        if lost_gate_factor <= 0 or tentative_gate_factor <= 0:
            raise ValueError("gate factors must be positive.")
        if birth_suppression_radius_m < 0:
            raise ValueError("birth_suppression_radius_m must be non-negative.")
        if primary_track_birth_scale <= 0:
            raise ValueError("primary_track_birth_scale must be positive.")
        if birth_suppression_weak_radius_scale < 1.0:
            raise ValueError("birth_suppression_weak_radius_scale must be >= 1.")
        if birth_suppression_score_ratio < 0 or birth_suppression_confidence_ratio < 0:
            raise ValueError("birth suppression ratios must be non-negative.")
        if birth_suppression_doppler_bins < 0:
            raise ValueError("birth_suppression_doppler_bins must be non-negative.")
        if birth_suppression_miss_tolerance < 0:
            raise ValueError("birth_suppression_miss_tolerance must be non-negative.")
        if primary_track_hold_frames < 0:
            raise ValueError("primary_track_hold_frames must be non-negative.")
        if lateral_deadband_m < 0 or lateral_deadband_range_scale < 0:
            raise ValueError("lateral deadband values must be non-negative.")
        if not (0.0 < lateral_smoothing_alpha <= 1.0):
            raise ValueError("lateral_smoothing_alpha must be in (0, 1].")
        if not (0.0 < lateral_velocity_damping <= 1.0):
            raise ValueError("lateral_velocity_damping must be in (0, 1].")
        if lateral_range_damping_start_m < 0 or lateral_range_damping_full_m < 0:
            raise ValueError("lateral range damping distances must be non-negative.")
        if lateral_range_damping_full_m < lateral_range_damping_start_m:
            raise ValueError("lateral_range_damping_full_m must be >= lateral_range_damping_start_m.")
        if not (0.0 < lateral_range_damping_min_alpha <= 1.0):
            raise ValueError("lateral_range_damping_min_alpha must be in (0, 1].")
        if track_line_projection_min_points < 2:
            raise ValueError("track_line_projection_min_points must be at least 2.")
        if track_line_projection_history_frames < track_line_projection_min_points:
            raise ValueError(
                "track_line_projection_history_frames must be >= track_line_projection_min_points."
            )
        if not (0.0 <= track_line_projection_blend <= 1.0):
            raise ValueError("track_line_projection_blend must be in [0, 1].")
        if track_line_projection_max_shift_m < 0:
            raise ValueError("track_line_projection_max_shift_m must be non-negative.")
        if not (0.0 < forward_smoothing_alpha <= 1.0):
            raise ValueError("forward_smoothing_alpha must be in (0, 1].")
        if not (0.0 < forward_velocity_damping <= 1.0):
            raise ValueError("forward_velocity_damping must be in (0, 1].")
        if not (0.0 <= measurement_follow_blend <= 1.0):
            raise ValueError("measurement_follow_blend must be in [0, 1].")
        if not (0.0 <= measurement_follow_min_quality <= 1.0):
            raise ValueError("measurement_follow_min_quality must be in [0, 1].")
        if measurement_follow_max_residual_m < 0:
            raise ValueError("measurement_follow_max_residual_m must be non-negative.")
        if not (0.0 <= local_remeasurement_blend <= 1.0):
            raise ValueError("local_remeasurement_blend must be in [0, 1].")
        if local_remeasurement_max_shift_m < 0:
            raise ValueError("local_remeasurement_max_shift_m must be non-negative.")
        if not (0.0 <= local_remeasurement_track_bias <= 1.0):
            raise ValueError("local_remeasurement_track_bias must be in [0, 1].")
        if not (0.0 < measurement_soft_gate_floor <= 1.0):
            raise ValueError("measurement_soft_gate_floor must be in (0, 1].")
        if measurement_soft_gate_start_m < 0 or measurement_soft_gate_full_m < 0:
            raise ValueError("measurement soft gate distances must be non-negative.")
        if measurement_soft_gate_full_m < measurement_soft_gate_start_m:
            raise ValueError("measurement_soft_gate_full_m must be >= measurement_soft_gate_start_m.")
        if measurement_soft_gate_range_scale < 0 or measurement_soft_gate_speed_scale < 0:
            raise ValueError("measurement soft gate scales must be non-negative.")
        if motion_direction_min_speed_m_s < 0:
            raise ValueError("motion_direction_min_speed_m_s must be non-negative.")
        if motion_direction_min_displacement_m < 0:
            raise ValueError("motion_direction_min_displacement_m must be non-negative.")
        if not (0.0 < motion_direction_max_angle_deg <= 180.0):
            raise ValueError("motion_direction_max_angle_deg must be in (0, 180].")
        if motion_direction_max_cross_m < 0 or motion_direction_cross_range_scale < 0:
            raise ValueError("motion direction cross gate values must be non-negative.")
        if max_object_count < 0:
            raise ValueError("max_object_count must be non-negative.")
        if expected_object_count is not None and int(expected_object_count) < 1:
            raise ValueError("expected_object_count must be positive or None.")
        if crossing_hold_frames < 0:
            raise ValueError("crossing_hold_frames must be non-negative.")

        kalman_filter, q_discrete_white_noise = _load_filterpy()
        self._KalmanFilter = kalman_filter
        self._QDiscreteWhiteNoise = q_discrete_white_noise

        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.range_measurement_scale = float(range_measurement_scale)
        self.confidence_measurement_scale = float(confidence_measurement_scale)
        self.angle_resolution_rad = float(angle_resolution_rad)
        self.lateral_measurement_scale = float(lateral_measurement_scale)
        self.forward_measurement_scale = float(forward_measurement_scale)
        self.association_gate = float(association_gate)
        self.doppler_center_bin = None if doppler_center_bin is None else int(doppler_center_bin)
        self.doppler_zero_guard_bins = int(doppler_zero_guard_bins)
        self.doppler_gate_bins = int(doppler_gate_bins)
        self.doppler_cost_weight = float(doppler_cost_weight)
        self.max_missed_frames = int(max_missed_frames)
        self.min_confirmed_hits = int(min_confirmed_hits)
        self.report_miss_tolerance = int(report_miss_tolerance)
        self.lost_gate_factor = float(lost_gate_factor)
        self.tentative_gate_factor = float(tentative_gate_factor)
        self.birth_suppression_radius_m = float(birth_suppression_radius_m)
        self.primary_track_birth_scale = float(primary_track_birth_scale)
        self.birth_suppression_weak_radius_scale = float(birth_suppression_weak_radius_scale)
        self.birth_suppression_score_ratio = float(birth_suppression_score_ratio)
        self.birth_suppression_confidence_ratio = float(birth_suppression_confidence_ratio)
        self.birth_suppression_doppler_bins = int(birth_suppression_doppler_bins)
        self.birth_suppression_miss_tolerance = int(birth_suppression_miss_tolerance)
        self.primary_track_hold_frames = int(primary_track_hold_frames)
        self.lateral_deadband_m = float(lateral_deadband_m)
        self.lateral_deadband_range_scale = float(lateral_deadband_range_scale)
        self.lateral_smoothing_alpha = float(lateral_smoothing_alpha)
        self.lateral_velocity_damping = float(lateral_velocity_damping)
        self.lateral_range_damping_enabled = bool(lateral_range_damping_enabled)
        self.lateral_range_damping_start_m = float(lateral_range_damping_start_m)
        self.lateral_range_damping_full_m = float(lateral_range_damping_full_m)
        self.lateral_range_damping_min_alpha = float(lateral_range_damping_min_alpha)
        self.track_line_projection_enabled = bool(track_line_projection_enabled)
        self.track_line_projection_min_points = int(track_line_projection_min_points)
        self.track_line_projection_history_frames = int(track_line_projection_history_frames)
        self.track_line_projection_blend = float(track_line_projection_blend)
        self.track_line_projection_max_shift_m = float(track_line_projection_max_shift_m)
        self.forward_smoothing_alpha = float(forward_smoothing_alpha)
        self.forward_velocity_damping = float(forward_velocity_damping)
        self.measurement_follow_enabled = bool(measurement_follow_enabled)
        self.measurement_follow_blend = float(measurement_follow_blend)
        self.measurement_follow_min_quality = float(measurement_follow_min_quality)
        self.measurement_follow_max_residual_m = float(measurement_follow_max_residual_m)
        self.local_remeasurement_enabled = bool(local_remeasurement_enabled)
        self.local_remeasurement_blend = float(local_remeasurement_blend)
        self.local_remeasurement_max_shift_m = float(local_remeasurement_max_shift_m)
        self.local_remeasurement_track_bias = float(local_remeasurement_track_bias)
        self.local_remeasurement_patch_bands = tuple(local_remeasurement_patch_bands or ())
        self.measurement_soft_gate_enabled = bool(measurement_soft_gate_enabled)
        self.measurement_soft_gate_floor = float(measurement_soft_gate_floor)
        self.measurement_soft_gate_start_m = float(measurement_soft_gate_start_m)
        self.measurement_soft_gate_full_m = float(measurement_soft_gate_full_m)
        self.measurement_soft_gate_range_scale = float(measurement_soft_gate_range_scale)
        self.measurement_soft_gate_speed_scale = float(measurement_soft_gate_speed_scale)
        self.motion_direction_gate_enabled = bool(motion_direction_gate_enabled)
        self.motion_direction_min_speed_m_s = float(motion_direction_min_speed_m_s)
        self.motion_direction_min_displacement_m = float(motion_direction_min_displacement_m)
        self.motion_direction_max_angle_deg = float(motion_direction_max_angle_deg)
        self.motion_direction_cos_threshold = float(cos(radians(motion_direction_max_angle_deg)))
        self.motion_direction_max_cross_m = float(motion_direction_max_cross_m)
        self.motion_direction_cross_range_scale = float(motion_direction_cross_range_scale)
        self.max_object_count = int(max_object_count)
        self.expected_object_count = (
            None if expected_object_count is None else int(expected_object_count)
        )
        self.crossing_hold_frames = int(crossing_hold_frames)

        self._tracks: List[_Track] = []
        self._next_track_id = 1
        self._last_frame_ts: Optional[float] = None
        self._primary_track_id: Optional[int] = None
        self._track_path_history: Dict[int, List[Tuple[float, float]]] = {}

    def _active_track_limit(self) -> int:
        if self.expected_object_count is not None:
            return int(self.expected_object_count)
        return int(self.max_object_count)

    def _active_track_count(self) -> int:
        return sum(
            1
            for track in self._tracks
            if track.state != TrackState.TENTATIVE
            or track.consecutive_hits >= self.min_confirmed_hits
        )

    def _track_keep_key(self, track: _Track) -> tuple:
        is_primary = int(track.track_id) == int(self._primary_track_id or -1)
        return (
            1 if is_primary else 0,
            self._track_state_rank(track),
            -int(track.misses),
            float(track.confidence),
            float(track.score),
            int(track.hits),
            int(track.consecutive_hits),
            int(track.age),
        )

    def _prune_excess_tracks_by_count(self) -> list[int]:
        limit = self._active_track_limit()
        if limit <= 0 or len(self._tracks) <= limit:
            return []
        ordered = sorted(self._tracks, key=self._track_keep_key, reverse=True)
        keep_ids = {int(track.track_id) for track in ordered[:limit]}
        deleted_ids = [
            int(track.track_id)
            for track in self._tracks
            if int(track.track_id) not in keep_ids
        ]
        if deleted_ids:
            self._tracks = [
                track for track in self._tracks if int(track.track_id) in keep_ids
            ]
        return sorted(deleted_ids)

    def _measurement_covariance(
        self,
        range_m: float,
        confidence: float,
        measurement_quality: float = 1.0,
    ) -> np.ndarray:
        extra_scale = 1.0 + (self.range_measurement_scale * max(float(range_m) - 0.5, 0.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        confidence_scale = max(
            0.45,
            1.0 - (self.confidence_measurement_scale * confidence),
        )
        measurement_quality = float(np.clip(measurement_quality, 0.05, 1.0))
        quality_scale = 1.0 / measurement_quality
        variance = self.measurement_var * min(extra_scale, 4.0) * confidence_scale * quality_scale
        lateral_variance = float(variance)
        if self.angle_resolution_rad > 0.0:
            lateral_step_m = max(float(range_m), 0.5) * self.angle_resolution_rad
            lateral_scale = 1.0 + (
                self.lateral_measurement_scale * min(lateral_step_m / 0.04, 5.0)
            )
            lateral_variance *= lateral_scale
        forward_variance = float(variance) * self.forward_measurement_scale
        return np.diag(np.asarray([lateral_variance, forward_variance], dtype=float))

    def _build_kf(self, measurement: dict):
        kf = self._KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array(
            [[measurement["x_m"]], [measurement["y_m"]], [0.0], [0.0]],
            dtype=float,
        )
        kf.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        kf.P = np.eye(4, dtype=float) * 20.0
        kf.R = self._measurement_covariance(
            measurement["range_m"],
            measurement["confidence"],
        )
        kf.Q = self._QDiscreteWhiteNoise(
            dim=2,
            dt=0.1,
            var=self.process_var,
            block_size=2,
            order_by_dim=False,
        )
        return kf

    def _track_by_id(self, track_id: Optional[int]) -> Optional[_Track]:
        if track_id is None:
            return None
        return next((track for track in self._tracks if track.track_id == track_id), None)

    @staticmethod
    def _track_state_rank(track: _Track) -> int:
        if track.state == TrackState.CONFIRMED:
            return 2
        if track.state == TrackState.LOST:
            return 1
        return 0

    @staticmethod
    def _track_is_weak_confirmed(track: _Track) -> bool:
        if track.state != TrackState.CONFIRMED:
            return False
        low_quality = float(track.confidence) < 0.08 and float(track.score) < 0.2
        residually_weak = (
            float(track.confidence) < 0.12
            and float(track.score) < 0.35
            and float(track.last_measurement_residual_m) >= 0.45
        )
        return bool(low_quality or residually_weak)

    def _primary_candidate_key(self, track: _Track) -> tuple:
        capped_hits = min(int(track.hits), int(self.min_confirmed_hits) + 4)
        residual_m = min(float(track.last_measurement_residual_m), 9.99)
        return (
            self._track_state_rank(track),
            -int(track.misses),
            float(track.confidence),
            float(track.score),
            float(track.last_measurement_quality),
            -residual_m,
            capped_hits,
            int(track.consecutive_hits),
            int(track.age),
        )

    def _primary_handoff_needed(self, current: _Track, candidate: _Track) -> bool:
        if int(current.track_id) == int(candidate.track_id):
            return False
        if current.state == TrackState.TENTATIVE:
            return True
        if current.misses > self.primary_track_hold_frames:
            return True
        if candidate.state == TrackState.TENTATIVE and candidate.hits < self.min_confirmed_hits:
            return False
        if not self._track_is_weak_confirmed(current):
            return False

        current_x = float(current.kf.x[0][0])
        current_y = float(current.kf.x[1][0])
        candidate_x = float(candidate.kf.x[0][0])
        candidate_y = float(candidate.kf.x[1][0])
        handoff_distance_m = float(hypot(candidate_x - current_x, candidate_y - current_y))
        current_range_m = float(hypot(current_x, current_y))
        handoff_limit_m = max(0.55, min(0.9, 0.35 + (0.12 * current_range_m)))
        if current.misses <= self.primary_track_hold_frames and handoff_distance_m > handoff_limit_m:
            return False

        candidate_strong = float(candidate.confidence) >= 0.25 or float(candidate.score) >= 1.0
        confidence_advantage = float(candidate.confidence) >= max(
            float(current.confidence) * 3.0,
            float(current.confidence) + 0.15,
        )
        score_advantage = float(candidate.score) >= max(
            float(current.score) * 3.0,
            float(current.score) + 0.5,
        )
        residual_advantage = (
            float(candidate.last_measurement_residual_m) + 0.15
            <= float(current.last_measurement_residual_m)
        ) or float(current.last_measurement_residual_m) >= 0.45
        return bool(candidate_strong and residual_advantage and (confidence_advantage or score_advantage))

    def _update_primary_track_id(self) -> None:
        candidates = [
            track
            for track in self._tracks
            if track.state != TrackState.TENTATIVE or track.hits >= self.min_confirmed_hits
        ]
        if not candidates:
            self._primary_track_id = None
            return

        best_track = max(candidates, key=self._primary_candidate_key)
        current_primary = self._track_by_id(self._primary_track_id)
        if (
            current_primary is not None
            and current_primary.state != TrackState.TENTATIVE
            and current_primary.misses <= self.primary_track_hold_frames
            and not self._primary_handoff_needed(current_primary, best_track)
        ):
            return

        self._primary_track_id = best_track.track_id

    def _birth_suppression_radius_for_track(self, track: _Track) -> float:
        radius = self.birth_suppression_radius_m
        if track.track_id == self._primary_track_id:
            radius *= self.primary_track_birth_scale
        return radius

    def _doppler_distance_bins(self, left_bin: int, right_bin: int) -> float:
        return abs(self._signed_doppler_bin(left_bin) - self._signed_doppler_bin(right_bin))

    def _weak_duplicate_birth(self, track: _Track, measurement: dict) -> bool:
        score_ratio_limit = float(self.birth_suppression_score_ratio)
        confidence_ratio_limit = float(self.birth_suppression_confidence_ratio)
        if score_ratio_limit <= 0.0 and confidence_ratio_limit <= 0.0:
            return False

        if self.birth_suppression_doppler_bins > 0:
            doppler_distance = self._doppler_distance_bins(track.doppler_bin, measurement["doppler_bin"])
            if doppler_distance > self.birth_suppression_doppler_bins:
                return False

        weak_score = False
        if score_ratio_limit > 0.0:
            score_ratio = float(measurement["score"]) / max(float(track.score), 1e-6)
            weak_score = weak_score or score_ratio <= score_ratio_limit
        if confidence_ratio_limit > 0.0:
            confidence_ratio = float(measurement["confidence"]) / max(float(track.confidence), 1e-6)
            weak_score = weak_score or confidence_ratio <= confidence_ratio_limit
        return bool(weak_score)

    def _track_is_stronger_duplicate_reference(self, track: _Track, reference: _Track) -> bool:
        if reference.track_id == track.track_id:
            return False
        if reference.misses > self.birth_suppression_miss_tolerance:
            return False
        if reference.state == TrackState.CONFIRMED:
            return True
        if reference.state == TrackState.LOST:
            return reference.hits >= track.hits
        if reference.hits > track.hits:
            return True
        if reference.hits < track.hits:
            return False
        return (
            reference.consecutive_hits,
            reference.confidence,
            reference.score,
            reference.age,
            -reference.track_id,
        ) >= (
            track.consecutive_hits,
            track.confidence,
            track.score,
            track.age,
            -track.track_id,
        )

    def _weak_duplicate_track(self, track: _Track, reference: _Track) -> bool:
        score_ratio_limit = float(self.birth_suppression_score_ratio)
        confidence_ratio_limit = float(self.birth_suppression_confidence_ratio)
        if score_ratio_limit <= 0.0 and confidence_ratio_limit <= 0.0:
            return track.hits <= reference.hits

        if self.birth_suppression_doppler_bins > 0:
            doppler_distance = self._doppler_distance_bins(track.doppler_bin, reference.doppler_bin)
            if doppler_distance > self.birth_suppression_doppler_bins:
                return False

        weak_score = False
        if score_ratio_limit > 0.0:
            score_ratio = float(track.score) / max(float(reference.score), 1e-6)
            weak_score = weak_score or score_ratio <= score_ratio_limit
        if confidence_ratio_limit > 0.0:
            confidence_ratio = float(track.confidence) / max(float(reference.confidence), 1e-6)
            weak_score = weak_score or confidence_ratio <= confidence_ratio_limit
        return bool(weak_score)

    def _should_prune_duplicate_tentative(self, track: _Track) -> bool:
        if track.state != TrackState.TENTATIVE or self.birth_suppression_radius_m <= 0:
            return False

        for reference in self._tracks:
            if not self._track_is_stronger_duplicate_reference(track, reference):
                continue

            radius = self._birth_suppression_radius_for_track(reference)
            if radius <= 0:
                continue

            dx = float(reference.kf.x[0][0]) - float(track.kf.x[0][0])
            dy = float(reference.kf.x[1][0]) - float(track.kf.x[1][0])
            distance_m = float(hypot(dx, dy))
            if distance_m <= radius:
                return True
            if (
                self.birth_suppression_weak_radius_scale > 1.0
                and distance_m <= radius * self.birth_suppression_weak_radius_scale
                and self._weak_duplicate_track(track, reference)
            ):
                return True

        return False

    def _should_prune_weak_confirmed_duplicate(self, track: _Track) -> bool:
        if self.birth_suppression_radius_m <= 0:
            return False
        if int(track.track_id) == int(self._primary_track_id or -1):
            return False
        if not self._track_is_weak_confirmed(track):
            return False

        for reference in self._tracks:
            if int(reference.track_id) == int(track.track_id):
                continue
            if reference.state != TrackState.CONFIRMED:
                continue
            if reference.misses > self.birth_suppression_miss_tolerance:
                continue
            if float(reference.confidence) < 0.25 and float(reference.score) < 1.0:
                continue
            if not self._weak_duplicate_track(track, reference):
                continue

            radius = self._birth_suppression_radius_for_track(reference)
            if radius <= 0:
                continue
            dx = float(reference.kf.x[0][0]) - float(track.kf.x[0][0])
            dy = float(reference.kf.x[1][0]) - float(track.kf.x[1][0])
            distance_m = float(hypot(dx, dy))
            expanded_radius = radius * max(float(self.birth_suppression_weak_radius_scale), 1.0)
            if distance_m <= expanded_radius:
                return True
            if abs(dx) <= radius and abs(dy) <= expanded_radius * 1.35:
                return True

        return False

    def _should_suppress_birth(self, measurement: dict) -> bool:
        limit = self._active_track_limit()
        if limit > 0 and self._active_track_count() >= limit:
            return True

        if self.birth_suppression_radius_m <= 0:
            return False

        for track in self._tracks:
            if track.misses > self.birth_suppression_miss_tolerance:
                continue
            if track.state == TrackState.TENTATIVE and track.consecutive_hits < self.min_confirmed_hits:
                continue

            radius = self._birth_suppression_radius_for_track(track)
            if radius <= 0:
                continue

            dx = float(track.kf.x[0][0]) - float(measurement["x_m"])
            dy = float(track.kf.x[1][0]) - float(measurement["y_m"])
            distance_m = float(hypot(dx, dy))
            if distance_m <= radius:
                return True
            if (
                self.birth_suppression_weak_radius_scale > 1.0
                and distance_m <= radius * self.birth_suppression_weak_radius_scale
                and self._weak_duplicate_birth(track, measurement)
            ):
                return True

        return False

    def _lateral_deadband_for_range(self, range_m: float) -> float:
        extra = max(float(range_m) - 0.5, 0.0) * self.lateral_deadband_range_scale
        return min(self.lateral_deadband_m + extra, 0.25)

    def _lateral_alpha_for_range(self, range_m: float) -> float:
        base_alpha = float(self.lateral_smoothing_alpha)
        if not self.lateral_range_damping_enabled:
            return base_alpha
        start_m = float(self.lateral_range_damping_start_m)
        full_m = float(self.lateral_range_damping_full_m)
        min_alpha = min(float(self.lateral_range_damping_min_alpha), base_alpha)
        if full_m <= start_m:
            return min_alpha if float(range_m) >= start_m else base_alpha
        t = (float(range_m) - start_m) / (full_m - start_m)
        t = float(np.clip(t, 0.0, 1.0))
        return (base_alpha * (1.0 - t)) + (min_alpha * t)

    def _stabilize_lateral_state(
        self,
        predicted_x: float,
        updated_x: float,
        updated_vx: float,
        range_m: float,
    ) -> tuple[float, float]:
        delta_x = float(updated_x) - float(predicted_x)
        deadband = self._lateral_deadband_for_range(range_m)
        if abs(delta_x) <= deadband:
            stabilized_x = float(predicted_x) + (delta_x * 0.15)
        else:
            stabilized_x = float(predicted_x) + (
                delta_x * self._lateral_alpha_for_range(range_m)
            )
        stabilized_vx = float(updated_vx) * self.lateral_velocity_damping
        return stabilized_x, stabilized_vx

    def _stabilize_forward_state(
        self,
        predicted_y: float,
        updated_y: float,
        updated_vy: float,
    ) -> tuple[float, float]:
        delta_y = float(updated_y) - float(predicted_y)
        stabilized_y = float(predicted_y) + (delta_y * self.forward_smoothing_alpha)
        stabilized_vy = float(updated_vy) * self.forward_velocity_damping
        return stabilized_y, stabilized_vy

    def _append_track_path_history(self, track_id: int, x_m: float, y_m: float) -> None:
        history = self._track_path_history.setdefault(int(track_id), [])
        history.append((float(x_m), float(y_m)))
        overflow = len(history) - int(self.track_line_projection_history_frames)
        if overflow > 0:
            del history[:overflow]

    def _line_project_track_state(
        self,
        track_id: int,
        x_m: float,
        y_m: float,
    ) -> tuple[float, float, float]:
        if not self.track_line_projection_enabled:
            return float(x_m), float(y_m), 0.0
        history = self._track_path_history.get(int(track_id), [])
        if len(history) < self.track_line_projection_min_points:
            return float(x_m), float(y_m), 0.0

        points = np.asarray(history[-self.track_line_projection_history_frames :], dtype=float)
        center = np.mean(points, axis=0)
        centered = points - center
        cov = centered.T @ centered
        if not np.all(np.isfinite(cov)):
            return float(x_m), float(y_m), 0.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_index = int(np.argmax(eigvals))
        if float(eigvals[major_index]) <= 1e-6:
            return float(x_m), float(y_m), 0.0
        direction = eigvecs[:, major_index]
        measurement = np.asarray([float(x_m), float(y_m)], dtype=float)
        projected = center + (np.dot(measurement - center, direction) * direction)
        shift_vec = projected - measurement
        shift_m = float(np.linalg.norm(shift_vec))
        if shift_m <= 1e-9:
            return float(x_m), float(y_m), 0.0

        requested_shift = shift_m * self.track_line_projection_blend
        applied_shift = min(requested_shift, self.track_line_projection_max_shift_m)
        if applied_shift <= 0:
            return float(x_m), float(y_m), 0.0
        adjusted = measurement + (shift_vec * (applied_shift / shift_m))
        return float(adjusted[0]), float(adjusted[1]), float(applied_shift)

    def _follow_measurement_state(
        self,
        x_m: float,
        y_m: float,
        measurement: dict,
        measurement_quality: float,
        residual_m: float,
    ) -> tuple[float, float, float, float]:
        if not self.measurement_follow_enabled or self.measurement_follow_blend <= 0.0:
            return float(x_m), float(y_m), 0.0, 0.0
        if float(measurement_quality) < self.measurement_follow_min_quality:
            return float(x_m), float(y_m), 0.0, 0.0
        if (
            self.measurement_follow_max_residual_m > 0.0
            and float(residual_m) > self.measurement_follow_max_residual_m
        ):
            return float(x_m), float(y_m), 0.0, 0.0

        target_x = float(measurement["x_m"])
        target_y = float(measurement["y_m"])
        if not (isfinite(target_x) and isfinite(target_y)):
            return float(x_m), float(y_m), 0.0, 0.0

        blend = float(self.measurement_follow_blend)
        adjusted_x = float(x_m) + ((target_x - float(x_m)) * blend)
        adjusted_y = float(y_m) + ((target_y - float(y_m)) * blend)
        shift_m = float(hypot(adjusted_x - float(x_m), adjusted_y - float(y_m)))
        return adjusted_x, adjusted_y, shift_m, blend

    def _local_remeasurement_patch_for_range(self, range_m: float) -> tuple[int, int, float]:
        range_radius_bins = 1
        angle_radius_bins = 2
        relative_floor = 0.55

        for band in self.local_remeasurement_patch_bands:
            try:
                r_min = float(band.get("r_min", 0.0))
                r_max = band.get("r_max")
            except (AttributeError, TypeError, ValueError):
                continue

            if float(range_m) < r_min:
                continue
            if r_max is not None and float(range_m) >= float(r_max):
                continue

            try:
                range_radius_bins = max(1, int(band.get("range_radius_bins", range_radius_bins)))
                angle_radius_bins = max(1, int(band.get("angle_radius_bins", angle_radius_bins)))
                relative_floor = float(band.get("relative_floor", relative_floor))
            except (TypeError, ValueError):
                pass
            break

        return (
            int(range_radius_bins),
            int(angle_radius_bins),
            float(np.clip(relative_floor, 0.0, 0.95)),
        )

    def _refine_measurement_near_track(
        self,
        track: _Track,
        measurement: dict,
        rai_map=None,
        runtime_config=None,
    ) -> dict:
        if (
            not self.local_remeasurement_enabled
            or self.local_remeasurement_blend <= 0.0
            or rai_map is None
            or runtime_config is None
        ):
            return measurement

        rai_array = np.asarray(rai_map, dtype=np.float64)
        if rai_array.ndim != 2 or rai_array.size == 0:
            return measurement

        try:
            range_axis = np.asarray(runtime_config.range_axis_m, dtype=np.float64)
            angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
        except AttributeError:
            return measurement
        if range_axis.size == 0 or angle_axis.size == 0:
            return measurement
        if range_axis.size < rai_array.shape[0] or angle_axis.size < rai_array.shape[1]:
            return measurement

        predicted_x = float(track.kf.x[0][0])
        predicted_y = float(track.kf.x[1][0])
        track_bias = self.local_remeasurement_track_bias
        seed_x = (float(measurement["x_m"]) * (1.0 - track_bias)) + (predicted_x * track_bias)
        seed_y = (float(measurement["y_m"]) * (1.0 - track_bias)) + (predicted_y * track_bias)
        seed_range_m = float(hypot(seed_x, seed_y))
        seed_angle_rad = float(atan2(seed_x, max(seed_y, 1e-6)))
        seed_range_bin = int(np.clip(_nearest_axis_bin(range_axis, seed_range_m), 0, rai_array.shape[0] - 1))
        seed_angle_bin = int(np.clip(_nearest_axis_bin(angle_axis, seed_angle_rad), 0, rai_array.shape[1] - 1))

        range_radius_bins, angle_radius_bins, relative_floor = self._local_remeasurement_patch_for_range(
            max(float(measurement["range_m"]), seed_range_m),
        )
        angle_lower = max(seed_angle_bin - angle_radius_bins, 0)
        angle_upper = min(seed_angle_bin + angle_radius_bins + 1, rai_array.shape[1])
        range_lower = max(seed_range_bin - range_radius_bins, 0)
        range_upper = min(seed_range_bin + range_radius_bins + 1, rai_array.shape[0])
        local_patch = rai_array[range_lower:range_upper, angle_lower:angle_upper]
        if local_patch.size == 0 or not np.any(np.isfinite(local_patch)):
            return measurement

        finite_patch = local_patch[np.isfinite(local_patch)]
        finite_patch = finite_patch[finite_patch > 0.0]
        if finite_patch.size == 0:
            return measurement

        angle_mask = np.ones_like(angle_axis, dtype=bool)
        angle_floor = float(np.quantile(finite_patch, 0.45))
        (
            refined_range_bin,
            refined_angle_bin,
            refined_range_m,
            _refined_angle_rad,
            refined_x_m,
            refined_y_m,
        ) = _refine_body_center_from_patch(
            rai_array,
            runtime_config,
            seed_range_bin,
            seed_angle_bin,
            angle_mask,
            angle_floor=angle_floor,
            range_radius_bins=range_radius_bins,
            angle_radius_bins=angle_radius_bins,
            relative_floor=relative_floor,
        )

        if not (np.isfinite(refined_x_m) and np.isfinite(refined_y_m)):
            return measurement

        shift_x = float(refined_x_m) - float(measurement["x_m"])
        shift_y = float(refined_y_m) - float(measurement["y_m"])
        shift_distance = float(hypot(shift_x, shift_y))
        max_shift_m = self.local_remeasurement_max_shift_m
        if max_shift_m > 0.0 and shift_distance > max_shift_m:
            scale = max_shift_m / max(shift_distance, 1e-6)
            refined_x_m = float(measurement["x_m"]) + (shift_x * scale)
            refined_y_m = float(measurement["y_m"]) + (shift_y * scale)
            refined_range_m = float(hypot(refined_x_m, refined_y_m))
            refined_range_bin = _nearest_axis_bin(range_axis, refined_range_m)
            refined_angle_bin = _nearest_axis_bin(angle_axis, atan2(refined_x_m, max(refined_y_m, 1e-6)))

        blend = self.local_remeasurement_blend
        blended_x_m = (float(measurement["x_m"]) * (1.0 - blend)) + (float(refined_x_m) * blend)
        blended_y_m = (float(measurement["y_m"]) * (1.0 - blend)) + (float(refined_y_m) * blend)
        blended_range_m = float(hypot(blended_x_m, blended_y_m))

        refined_measurement = dict(measurement)
        refined_measurement["x_m"] = blended_x_m
        refined_measurement["y_m"] = blended_y_m
        refined_measurement["range_m"] = blended_range_m
        refined_measurement["rdi_peak"] = float(measurement["rdi_peak"])
        refined_measurement["rai_peak"] = float(
            max(
                float(measurement["rai_peak"]),
                float(rai_array[
                    int(np.clip(refined_range_bin, 0, rai_array.shape[0] - 1)),
                    int(np.clip(refined_angle_bin, 0, rai_array.shape[1] - 1)),
                ]),
            )
        )
        return refined_measurement

    @staticmethod
    def _measurement_from_detection(detection) -> dict:
        confidence = float(np.clip(detection.score / 3.0, 0.0, 1.0))
        return {
            "x_m": float(detection.x_m),
            "y_m": float(detection.y_m),
            "range_m": float(detection.range_m),
            "doppler_bin": int(detection.doppler_bin),
            "rdi_peak": float(detection.rdi_peak),
            "rai_peak": float(detection.rai_peak),
            "score": float(detection.score),
            "confidence": confidence,
        }

    def _measurement_soft_gate_thresholds(
        self,
        track: _Track,
        measurement: dict,
    ) -> tuple[float, float]:
        range_extra = self.measurement_soft_gate_range_scale * max(
            float(measurement["range_m"]) - 0.5,
            0.0,
        )
        speed_m_s = float(hypot(track.kf.x[2][0], track.kf.x[3][0]))
        speed_extra = min(speed_m_s * self.measurement_soft_gate_speed_scale, 0.18)

        start_m = self.measurement_soft_gate_start_m + range_extra + speed_extra
        full_m = self.measurement_soft_gate_full_m + (range_extra * 1.8) + (speed_extra * 1.6)

        if track.state == TrackState.LOST:
            start_m *= 1.15
            full_m *= 1.25
        elif track.state == TrackState.TENTATIVE:
            start_m *= 0.9
            full_m *= 0.9

        return float(start_m), float(max(full_m, start_m + 0.05))

    def _measurement_update_quality(
        self,
        track: _Track,
        measurement: dict,
        predicted_x: float,
        predicted_y: float,
    ) -> tuple[float, float]:
        residual_m = float(
            hypot(
                float(measurement["x_m"]) - float(predicted_x),
                float(measurement["y_m"]) - float(predicted_y),
            )
        )
        if not self.measurement_soft_gate_enabled:
            return 1.0, residual_m

        start_m, full_m = self._measurement_soft_gate_thresholds(track, measurement)
        if residual_m <= start_m:
            return 1.0, residual_m
        if residual_m >= full_m:
            return self.measurement_soft_gate_floor, residual_m

        progress = (residual_m - start_m) / max(full_m - start_m, 1e-6)
        quality = 1.0 - ((1.0 - self.measurement_soft_gate_floor) * progress)
        return float(np.clip(quality, self.measurement_soft_gate_floor, 1.0)), residual_m

    def _signed_doppler_bin(self, doppler_bin: int) -> float:
        if self.doppler_center_bin is None:
            return float(doppler_bin)
        return float(int(doppler_bin) - self.doppler_center_bin)

    def _doppler_consistency_cost(self, track: _Track, measurement: dict) -> float:
        if self.doppler_gate_bins <= 0 or self.doppler_cost_weight <= 0:
            return 0.0

        track_signed = self._signed_doppler_bin(track.doppler_bin)
        measurement_signed = self._signed_doppler_bin(measurement["doppler_bin"])
        track_is_near_zero = abs(track_signed) <= self.doppler_zero_guard_bins
        measurement_is_near_zero = abs(measurement_signed) <= self.doppler_zero_guard_bins

        if track_is_near_zero and measurement_is_near_zero:
            return 0.0

        doppler_delta = abs(track_signed - measurement_signed)
        if doppler_delta > self.doppler_gate_bins:
            return np.inf

        normalized_delta = doppler_delta / max(float(self.doppler_gate_bins), 1.0)
        penalty = normalized_delta * normalized_delta
        sign_mismatch = (
            (track_signed * measurement_signed) < 0.0
            and not track_is_near_zero
            and not measurement_is_near_zero
        )
        if sign_mismatch:
            penalty += 1.0

        return self.doppler_cost_weight * penalty

    def _motion_direction_reject(self, track: _Track, measurement: dict) -> bool:
        if not self.motion_direction_gate_enabled:
            return False
        if track.state != TrackState.CONFIRMED:
            return False
        if track.consecutive_hits < self.min_confirmed_hits:
            return False

        vx = float(track.kf.x[2][0])
        vy = float(track.kf.x[3][0])
        speed_m_s = float(hypot(vx, vy))
        if speed_m_s < self.motion_direction_min_speed_m_s:
            return False

        dx = float(measurement["x_m"]) - float(track.kf.x[0][0])
        dy = float(measurement["y_m"]) - float(track.kf.x[1][0])
        displacement_m = float(hypot(dx, dy))
        if displacement_m < self.motion_direction_min_displacement_m:
            return False

        ux = vx / max(speed_m_s, 1e-6)
        uy = vy / max(speed_m_s, 1e-6)
        alignment = ((dx * ux) + (dy * uy)) / max(displacement_m, 1e-6)
        if alignment < self.motion_direction_cos_threshold:
            return True

        cross_m = abs((dx * uy) - (dy * ux))
        range_m = float(measurement.get("range_m", displacement_m))
        cross_limit_m = (
            self.motion_direction_max_cross_m
            + self.motion_direction_cross_range_scale * max(range_m - 0.5, 0.0)
        )
        mostly_sideways = alignment < 0.35
        return bool(mostly_sideways and cross_m > cross_limit_m)

    def _mahalanobis_sq(self, track: _Track, measurement: dict) -> float:
        z = np.array([[measurement["x_m"]], [measurement["y_m"]]], dtype=float)
        innovation = z - (track.kf.H @ track.kf.x)
        innovation_cov = (
            track.kf.H @ track.kf.P @ track.kf.H.T
            + self._measurement_covariance(
                measurement["range_m"],
                measurement["confidence"],
            )
        )
        try:
            solved = np.linalg.solve(innovation_cov, innovation)
        except np.linalg.LinAlgError:
            return np.inf
        return float((innovation.T @ solved)[0, 0])

    def _compute_dt(self, frame_ts: Optional[float]) -> float:
        if frame_ts is None or self._last_frame_ts is None:
            return 0.1

        delta = frame_ts - self._last_frame_ts
        if delta <= 0:
            return 0.1
        return max(0.03, min(0.5, delta))

    def _predict(self, dt: float) -> None:
        q_matrix = self._QDiscreteWhiteNoise(
            dim=2,
            dt=dt,
            var=self.process_var,
            block_size=2,
            order_by_dim=False,
        )

        for track in self._tracks:
            track.kf.F[0, 2] = dt
            track.kf.F[1, 3] = dt
            track.kf.Q = q_matrix
            track.kf.predict()
            track.age += 1
            track.misses += 1
            track.confidence *= 0.96
            track.score *= 0.97

    def _run_hungarian(
        self,
        measurements: List[dict],
        track_indices: List[int],
        measurement_indices: List[int],
        gate: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_indices or not measurement_indices:
            return [], track_indices[:], measurement_indices[:]

        invalid_cost = 1e9
        cost_matrix = np.full(
            (len(track_indices), len(measurement_indices)),
            invalid_cost,
            dtype=float,
        )

        for row, track_index in enumerate(track_indices):
            for col, measurement_index in enumerate(measurement_indices):
                track = self._tracks[track_index]
                measurement = measurements[measurement_index]
                if self._motion_direction_reject(track, measurement):
                    continue
                cost = self._mahalanobis_sq(
                    track,
                    measurement,
                )
                doppler_cost = self._doppler_consistency_cost(track, measurement)
                combined_cost = cost + doppler_cost
                if combined_cost <= gate:
                    cost_matrix[row, col] = combined_cost

        row_ind, col_ind = _linear_sum_assignment(cost_matrix)

        pairs = []
        used_rows = set()
        used_cols = set()
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] >= invalid_cost:
                continue
            pairs.append((track_indices[row], measurement_indices[col]))
            used_rows.add(int(row))
            used_cols.add(int(col))

        unmatched_tracks = [
            track_indices[row]
            for row in range(len(track_indices))
            if row not in used_rows
        ]
        unmatched_measurements = [
            measurement_indices[col]
            for col in range(len(measurement_indices))
            if col not in used_cols
        ]
        return pairs, unmatched_tracks, unmatched_measurements

    def _associate(
        self,
        measurements: List[dict],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self._tracks or not measurements:
            return [], list(range(len(self._tracks))), list(range(len(measurements)))

        all_measurements = list(range(len(measurements)))
        confirmed_indices = [
            index for index, track in enumerate(self._tracks)
            if track.state == TrackState.CONFIRMED
        ]
        lost_indices = [
            index for index, track in enumerate(self._tracks)
            if track.state == TrackState.LOST
        ]
        tentative_indices = [
            index for index, track in enumerate(self._tracks)
            if track.state == TrackState.TENTATIVE
        ]

        reacquire_gate = self.association_gate * self.lost_gate_factor
        tentative_gate = self.association_gate * self.tentative_gate_factor

        pairs1, unmatched_confirmed, remaining_measurements = self._run_hungarian(
            measurements,
            confirmed_indices,
            all_measurements,
            self.association_gate,
        )
        pairs2, unmatched_confirmed, remaining_measurements = self._run_hungarian(
            measurements,
            unmatched_confirmed,
            remaining_measurements,
            reacquire_gate,
        )
        pairs3, unmatched_lost, remaining_measurements = self._run_hungarian(
            measurements,
            lost_indices,
            remaining_measurements,
            reacquire_gate,
        )
        pairs4, unmatched_tentative, birth_measurements = self._run_hungarian(
            measurements,
            tentative_indices,
            remaining_measurements,
            tentative_gate,
        )

        return (
            pairs1 + pairs2 + pairs3 + pairs4,
            unmatched_confirmed + unmatched_lost + unmatched_tentative,
            birth_measurements,
        )

    def _track_to_estimate(self, track: _Track) -> TrackEstimate:
        x_m = float(track.kf.x[0][0])
        y_m = float(track.kf.x[1][0])
        vx_m_s = float(track.kf.x[2][0])
        vy_m_s = float(track.kf.x[3][0])
        range_m = float(hypot(x_m, y_m))
        angle_deg = float(degrees(atan2(x_m, max(y_m, 1e-6))))
        return TrackEstimate(
            track_id=track.track_id,
            x_m=x_m,
            y_m=y_m,
            vx_m_s=vx_m_s,
            vy_m_s=vy_m_s,
            range_m=range_m,
            angle_deg=angle_deg,
            doppler_bin=track.doppler_bin,
            rdi_peak=float(track.rdi_peak),
            rai_peak=float(track.rai_peak),
            score=float(track.score),
            confidence=float(track.confidence),
            age=track.age,
            hits=track.hits,
            misses=track.misses,
            measurement_quality=float(track.last_measurement_quality),
            measurement_residual_m=float(track.last_measurement_residual_m),
            is_primary=bool(track.track_id == self._primary_track_id),
        )

    @staticmethod
    def _trace_measurement(measurement: dict) -> dict:
        x_m = float(measurement["x_m"])
        y_m = float(measurement["y_m"])
        angle_deg = float(degrees(atan2(x_m, max(y_m, 1e-6))))
        return {
            "x_m": round(x_m, 4),
            "y_m": round(y_m, 4),
            "range_m": round(float(measurement["range_m"]), 4),
            "angle_deg": round(angle_deg, 3),
            "doppler_bin": int(measurement["doppler_bin"]),
            "score": round(float(measurement["score"]), 4),
            "confidence": round(float(measurement["confidence"]), 4),
            "rdi_peak": round(float(measurement["rdi_peak"]), 4),
            "rai_peak": round(float(measurement["rai_peak"]), 4),
        }

    @staticmethod
    def _trace_track(track: _Track) -> dict:
        return {
            "track_id": int(track.track_id),
            "state": track.state.name.lower(),
            "x_m": round(float(track.kf.x[0][0]), 4),
            "y_m": round(float(track.kf.x[1][0]), 4),
            "vx_m_s": round(float(track.kf.x[2][0]), 4),
            "vy_m_s": round(float(track.kf.x[3][0]), 4),
            "confidence": round(float(track.confidence), 4),
            "score": round(float(track.score), 4),
            "age": int(track.age),
            "hits": int(track.hits),
            "misses": int(track.misses),
            "consecutive_hits": int(track.consecutive_hits),
        }

    def update(
        self,
        detections,
        frame_ts: Optional[float] = None,
        allow_track_birth: bool = True,
        rai_map=None,
        runtime_config=None,
        trace=None,
    ):
        measurements = [
            self._measurement_from_detection(detection)
            for detection in detections
        ]
        dt = self._compute_dt(frame_ts)
        if trace is not None:
            trace.clear()
            trace.update(
                {
                    "trace_version": 1,
                    "dt_s": round(float(dt), 4),
                    "tracking_policy": {
                        "max_object_count": int(self.max_object_count),
                        "expected_object_count": self.expected_object_count,
                        "crossing_hold_frames": int(self.crossing_hold_frames),
                        "active_track_limit": int(self._active_track_limit()),
                    },
                    "input_detection_count": len(detections),
                    "measurement_count": len(measurements),
                    "allow_track_birth": bool(allow_track_birth),
                    "measurements": [self._trace_measurement(item) for item in measurements[:12]],
                    "tracks_before_predict": [self._trace_track(track) for track in self._tracks[:12]],
                }
            )
        self._predict(dt)
        if trace is not None:
            trace["kalman_prediction"] = {
                "track_count": len(self._tracks),
                "tracks": [self._trace_track(track) for track in self._tracks[:12]],
            }
        if frame_ts is not None:
            self._last_frame_ts = frame_ts

        matched_pairs, unmatched_tracks, unmatched_measurements = self._associate(measurements)
        if trace is not None:
            trace["association"] = {
                "matched_count": len(matched_pairs),
                "unmatched_track_count": len(unmatched_tracks),
                "unmatched_measurement_count": len(unmatched_measurements),
                "pairs": [
                    {
                        "track_id": int(self._tracks[track_index].track_id),
                        "measurement_index": int(measurement_index),
                        "measurement": self._trace_measurement(measurements[measurement_index]),
                    }
                    for track_index, measurement_index in matched_pairs[:12]
                ],
                "unmatched_track_ids": [
                    int(self._tracks[index].track_id)
                    for index in unmatched_tracks[:12]
                ],
                "unmatched_measurement_indices": [int(index) for index in unmatched_measurements[:12]],
            }

        update_events = []
        for track_index, measurement_index in matched_pairs:
            track = self._tracks[track_index]
            measurement = measurements[measurement_index]

            previous_state = track.state
            predicted_x = float(track.kf.x[0][0])
            predicted_y = float(track.kf.x[1][0])
            measurement = self._refine_measurement_near_track(
                track,
                measurement,
                rai_map=rai_map,
                runtime_config=runtime_config,
            )
            measurement_quality, measurement_residual_m = self._measurement_update_quality(
                track,
                measurement,
                predicted_x=predicted_x,
                predicted_y=predicted_y,
            )
            z = np.array([[measurement["x_m"]], [measurement["y_m"]]], dtype=float)
            track.kf.R = self._measurement_covariance(
                measurement["range_m"],
                measurement["confidence"],
                measurement_quality=measurement_quality,
            )
            track.kf.update(z)
            stabilized_x, stabilized_vx = self._stabilize_lateral_state(
                predicted_x=predicted_x,
                updated_x=float(track.kf.x[0][0]),
                updated_vx=float(track.kf.x[2][0]),
                range_m=measurement["range_m"],
            )
            stabilized_y, stabilized_vy = self._stabilize_forward_state(
                predicted_y=predicted_y,
                updated_y=float(track.kf.x[1][0]),
                updated_vy=float(track.kf.x[3][0]),
            )
            projected_x, projected_y, line_projection_shift_m = self._line_project_track_state(
                int(track.track_id),
                stabilized_x,
                stabilized_y,
            )
            followed_x, followed_y, measurement_follow_shift_m, measurement_follow_blend = (
                self._follow_measurement_state(
                    projected_x,
                    projected_y,
                    measurement,
                    measurement_quality,
                    measurement_residual_m,
                )
            )
            track.kf.x[0][0] = followed_x
            track.kf.x[1][0] = followed_y
            track.kf.x[2][0] = stabilized_vx
            track.kf.x[3][0] = stabilized_vy
            self._append_track_path_history(
                int(track.track_id),
                float(track.kf.x[0][0]),
                float(track.kf.x[1][0]),
            )
            track.hits += 1
            track.consecutive_hits += 1
            track.misses = 0
            if frame_ts is not None:
                track.last_update_ts = frame_ts
            quality_weight = 0.55 + (0.45 * measurement_quality)
            track.confidence = (
                (0.7 * track.confidence)
                + (0.3 * measurement["confidence"] * quality_weight)
            )
            track.score = (
                (0.65 * track.score)
                + (0.35 * measurement["score"] * quality_weight)
            )
            track.doppler_bin = int(measurement["doppler_bin"])
            track.rdi_peak = float(measurement["rdi_peak"])
            track.rai_peak = float(measurement["rai_peak"])
            track.last_measurement_quality = float(measurement_quality)
            track.last_measurement_residual_m = float(measurement_residual_m)
            if trace is not None and len(update_events) < 12:
                update_events.append(
                    {
                        "track_id": int(track.track_id),
                        "measurement_index": int(measurement_index),
                        "previous_state": previous_state.name.lower(),
                        "measurement_quality": round(float(measurement_quality), 4),
                        "measurement_residual_m": round(float(measurement_residual_m), 4),
                        "line_projection_shift_m": round(float(line_projection_shift_m), 4),
                        "measurement_follow_shift_m": round(float(measurement_follow_shift_m), 4),
                        "measurement_follow_blend": round(float(measurement_follow_blend), 4),
                        "updated_track": self._trace_track(track),
                    }
                )

            healthy_confirmed = (
                previous_state == TrackState.CONFIRMED
                and (track.confidence >= 0.08 or track.score >= 0.2)
            )
            if previous_state == TrackState.CONFIRMED:
                keep_primary = int(track.track_id) == int(self._primary_track_id or -1)
                track.state = (
                    TrackState.CONFIRMED
                    if healthy_confirmed or keep_primary
                    else TrackState.TENTATIVE
                )
            elif previous_state == TrackState.LOST or track.consecutive_hits >= self.min_confirmed_hits:
                track.state = TrackState.CONFIRMED
            else:
                track.state = TrackState.TENTATIVE
        if trace is not None:
            trace["kalman_update"] = {
                "updated_count": len(update_events),
                "events": update_events,
            }

        missed_events = []
        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.consecutive_hits = 0
            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
            if trace is not None and len(missed_events) < 12:
                missed_events.append(self._trace_track(track))

        self._update_primary_track_id()

        birth_events = []
        suppressed_births = []
        if allow_track_birth:
            for measurement_index in unmatched_measurements:
                measurement = measurements[measurement_index]
                if self._should_suppress_birth(measurement):
                    if trace is not None and len(suppressed_births) < 12:
                        suppressed_births.append(
                            {
                                "measurement_index": int(measurement_index),
                                "measurement": self._trace_measurement(measurement),
                            }
                        )
                    continue
                kf = self._build_kf(measurement)
                new_track = _Track(
                    track_id=self._next_track_id,
                    kf=kf,
                    age=1,
                    hits=1,
                    misses=0,
                    last_update_ts=frame_ts if frame_ts is not None else 0.0,
                    confidence=float(measurement["confidence"]),
                    score=float(measurement["score"]),
                    state=TrackState.CONFIRMED if self.min_confirmed_hits <= 1 else TrackState.TENTATIVE,
                    consecutive_hits=1,
                    doppler_bin=int(measurement["doppler_bin"]),
                    rdi_peak=float(measurement["rdi_peak"]),
                    rai_peak=float(measurement["rai_peak"]),
                    last_measurement_quality=1.0,
                    last_measurement_residual_m=0.0,
                )
                self._tracks.append(new_track)
                self._append_track_path_history(
                    int(new_track.track_id),
                    float(new_track.kf.x[0][0]),
                    float(new_track.kf.x[1][0]),
                )
                if trace is not None and len(birth_events) < 12:
                    birth_events.append(self._trace_track(new_track))
                self._next_track_id += 1
        elif trace is not None:
            suppressed_births.extend(
                {
                    "measurement_index": int(index),
                    "reason": "birth_not_allowed",
                    "measurement": self._trace_measurement(measurements[index]),
                }
                for index in unmatched_measurements[:12]
            )

        before_prune_ids = {int(track.track_id) for track in self._tracks}
        duplicate_confirmed_ids = [
            int(track.track_id)
            for track in self._tracks
            if self._should_prune_weak_confirmed_duplicate(track)
        ]
        duplicate_tentative_ids = [
            int(track.track_id)
            for track in self._tracks
            if self._should_prune_duplicate_tentative(track)
        ]
        duplicate_track_ids = sorted(set(duplicate_confirmed_ids) | set(duplicate_tentative_ids))
        if duplicate_track_ids:
            duplicate_track_id_set = set(duplicate_track_ids)
            self._tracks = [
                track for track in self._tracks
                if int(track.track_id) not in duplicate_track_id_set
            ]
        count_pruned_track_ids = self._prune_excess_tracks_by_count()
        self._tracks = [
            track for track in self._tracks
            if not (track.state == TrackState.TENTATIVE and track.misses > 1)
            and track.misses <= self.max_missed_frames
        ]
        after_prune_ids = {int(track.track_id) for track in self._tracks}
        deleted_ids = sorted((before_prune_ids - after_prune_ids) | set(count_pruned_track_ids))

        self._update_primary_track_id()

        confirmed_tracks = []
        tentative_tracks = []
        for track in self._tracks:
            estimate = self._track_to_estimate(track)
            if track.state == TrackState.TENTATIVE:
                tentative_tracks.append(estimate)
                continue
            if track.misses > self.report_miss_tolerance:
                continue
            confirmed_tracks.append(estimate)

        if trace is not None:
            trace["track_lifecycle"] = {
                "missed_tracks": missed_events,
                "births": birth_events,
                "suppressed_births": suppressed_births[:12],
                "duplicate_confirmed_deleted_ids": duplicate_confirmed_ids,
                "duplicate_tentative_deleted_ids": duplicate_tentative_ids,
                "count_pruned_track_ids": count_pruned_track_ids,
                "deleted_track_ids": deleted_ids,
                "primary_track_id": self._primary_track_id,
                "tracks_after_prune": [self._trace_track(track) for track in self._tracks[:12]],
            }
            trace["display_output"] = {
                "confirmed_count": len(confirmed_tracks),
                "tentative_count": len(tentative_tracks),
                "confirmed_track_ids": [int(track.track_id) for track in confirmed_tracks],
                "tentative_track_ids": [int(track.track_id) for track in tentative_tracks],
            }

        return confirmed_tracks, tentative_tracks
