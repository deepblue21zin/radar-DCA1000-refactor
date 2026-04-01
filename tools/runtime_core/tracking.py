"""Lightweight multi-target Kalman tracker for radar cluster centroids."""

from dataclasses import dataclass
from enum import Enum, auto
from math import atan2, degrees, hypot
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:
    _scipy_linear_sum_assignment = None


class _SimpleKalmanFilter:
    """Minimal linear Kalman filter fallback."""

    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1), dtype=float)
        self.F = np.eye(dim_x, dtype=float)
        self.H = np.zeros((dim_z, dim_x), dtype=float)
        self.P = np.eye(dim_x, dtype=float)
        self.Q = np.eye(dim_x, dtype=float)
        self.R = np.eye(dim_z, dtype=float)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - (self.H @ self.x)
        pht = self.P @ self.H.T
        s = self.H @ pht + self.R
        try:
            k = np.linalg.solve(s, pht.T).T
        except np.linalg.LinAlgError:
            k = pht @ np.linalg.pinv(s)
        self.x = self.x + (k @ y)
        identity = np.eye(self.dim_x, dtype=float)
        kh = k @ self.H
        self.P = (identity - kh) @ self.P @ (identity - kh).T + k @ self.R @ k.T
        return self.x


def _fallback_q_discrete_white_noise(
    dim: int,
    dt: float = 1.0,
    var: float = 1.0,
    block_size: int = 1,
    order_by_dim: bool = True,
) -> np.ndarray:
    if dim != 2:
        raise NotImplementedError("Fallback Q builder only supports dim=2.")

    q = np.array(
        [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
        dtype=float,
    ) * float(var)

    if block_size == 1:
        return q
    if block_size < 1:
        raise ValueError("block_size must be positive.")

    if order_by_dim:
        return np.kron(np.eye(block_size, dtype=float), q)
    return np.kron(q, np.eye(block_size, dtype=float))


def _load_filterpy():
    """Import filterpy, falling back to a local linear KF implementation."""
    try:
        from filterpy.common import Q_discrete_white_noise
        from filterpy.kalman import KalmanFilter
        return KalmanFilter, Q_discrete_white_noise
    except ImportError:
        vendor_root = Path(__file__).resolve().parents[1] / "filterpy-master"
        if vendor_root.exists() and str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        try:
            from filterpy.common import Q_discrete_white_noise
            from filterpy.kalman import KalmanFilter
            return KalmanFilter, Q_discrete_white_noise
        except ImportError:
            return _SimpleKalmanFilter, _fallback_q_discrete_white_noise


def _hungarian_fallback(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy assignment fallback for rectangular cost matrices."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional.")
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transposed = False
    rows, cols = cost.shape
    if rows > cols:
        cost = cost.T
        rows, cols = cost.shape
        transposed = True

    u = np.zeros(rows + 1, dtype=float)
    v = np.zeros(cols + 1, dtype=float)
    p = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)

    for row in range(1, rows + 1):
        p[0] = row
        col0 = 0
        minv = np.full(cols + 1, np.inf, dtype=float)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                cur = cost[row0 - 1, col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    row_ind = []
    col_ind = []
    for col in range(1, cols + 1):
        if p[col] != 0:
            row_ind.append(p[col] - 1)
            col_ind.append(col - 1)

    row_ind_array = np.asarray(row_ind, dtype=int)
    col_ind_array = np.asarray(col_ind, dtype=int)
    order = np.argsort(row_ind_array)
    row_ind_array = row_ind_array[order]
    col_ind_array = col_ind_array[order]

    if transposed:
        return col_ind_array, row_ind_array
    return row_ind_array, col_ind_array


def _linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)
    return _hungarian_fallback(cost_matrix)


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()


@dataclass
class TrackEstimate:
    track_id: int
    x_m: float
    y_m: float
    vx_m_s: float
    vy_m_s: float
    range_m: float
    angle_deg: float
    doppler_bin: int
    rdi_peak: float
    rai_peak: float
    score: float
    confidence: float
    age: int
    hits: int
    misses: int


@dataclass
class _Track:
    track_id: int
    kf: object
    age: int
    hits: int
    misses: int
    last_update_ts: float
    confidence: float
    score: float
    state: TrackState
    consecutive_hits: int
    doppler_bin: int
    rdi_peak: float
    rai_peak: float


class MultiTargetTracker:
    def __init__(
        self,
        process_var=1.0,
        measurement_var=0.4,
        range_measurement_scale=0.0,
        confidence_measurement_scale=0.0,
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
    ):
        if process_var <= 0:
            raise ValueError("process_var must be positive.")
        if measurement_var <= 0:
            raise ValueError("measurement_var must be positive.")
        if range_measurement_scale < 0:
            raise ValueError("range_measurement_scale must be non-negative.")
        if confidence_measurement_scale < 0:
            raise ValueError("confidence_measurement_scale must be non-negative.")
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

        kalman_filter, q_discrete_white_noise = _load_filterpy()
        self._KalmanFilter = kalman_filter
        self._QDiscreteWhiteNoise = q_discrete_white_noise

        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.range_measurement_scale = float(range_measurement_scale)
        self.confidence_measurement_scale = float(confidence_measurement_scale)
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

        self._tracks: List[_Track] = []
        self._next_track_id = 1
        self._last_frame_ts: Optional[float] = None

    def _measurement_covariance(self, range_m: float, confidence: float) -> np.ndarray:
        extra_scale = 1.0 + (self.range_measurement_scale * max(float(range_m) - 0.5, 0.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        confidence_scale = max(
            0.45,
            1.0 - (self.confidence_measurement_scale * confidence),
        )
        variance = self.measurement_var * min(extra_scale, 4.0) * confidence_scale
        return np.eye(2, dtype=float) * variance

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

    @staticmethod
    def _track_to_estimate(track: _Track) -> TrackEstimate:
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
        )

    def update(
        self,
        detections,
        frame_ts: Optional[float] = None,
        allow_track_birth: bool = True,
    ):
        measurements = [
            self._measurement_from_detection(detection)
            for detection in detections
        ]
        dt = self._compute_dt(frame_ts)
        self._predict(dt)
        if frame_ts is not None:
            self._last_frame_ts = frame_ts

        matched_pairs, unmatched_tracks, unmatched_measurements = self._associate(measurements)

        for track_index, measurement_index in matched_pairs:
            track = self._tracks[track_index]
            measurement = measurements[measurement_index]
            z = np.array([[measurement["x_m"]], [measurement["y_m"]]], dtype=float)

            previous_state = track.state
            track.kf.R = self._measurement_covariance(
                measurement["range_m"],
                measurement["confidence"],
            )
            track.kf.update(z)
            track.hits += 1
            track.consecutive_hits += 1
            track.misses = 0
            if frame_ts is not None:
                track.last_update_ts = frame_ts
            track.confidence = (0.7 * track.confidence) + (0.3 * measurement["confidence"])
            track.score = (0.65 * track.score) + (0.35 * measurement["score"])
            track.doppler_bin = int(measurement["doppler_bin"])
            track.rdi_peak = float(measurement["rdi_peak"])
            track.rai_peak = float(measurement["rai_peak"])

            if previous_state == TrackState.LOST or track.consecutive_hits >= self.min_confirmed_hits:
                track.state = TrackState.CONFIRMED
            else:
                track.state = TrackState.TENTATIVE

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.consecutive_hits = 0
            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

        if allow_track_birth:
            for measurement_index in unmatched_measurements:
                measurement = measurements[measurement_index]
                kf = self._build_kf(measurement)
                self._tracks.append(
                    _Track(
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
                    )
                )
                self._next_track_id += 1

        self._tracks = [
            track for track in self._tracks
            if not (track.state == TrackState.TENTATIVE and track.misses > 1)
            and track.misses <= self.max_missed_frames
        ]

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

        return confirmed_tracks, tentative_tracks
