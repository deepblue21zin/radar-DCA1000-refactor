from pathlib import Path
import sys

import numpy as np


class SimpleKalmanFilter:
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


def fallback_q_discrete_white_noise(
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


def load_filterpy():
    """Import filterpy, falling back to a local linear KF implementation."""
    try:
        from filterpy.common import Q_discrete_white_noise
        from filterpy.kalman import KalmanFilter
        return KalmanFilter, Q_discrete_white_noise
    except ImportError:
        vendor_root = Path(__file__).resolve().parents[2] / "filterpy-master"
        if vendor_root.exists() and str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        try:
            from filterpy.common import Q_discrete_white_noise
            from filterpy.kalman import KalmanFilter
            return KalmanFilter, Q_discrete_white_noise
        except ImportError:
            return SimpleKalmanFilter, fallback_q_discrete_white_noise
