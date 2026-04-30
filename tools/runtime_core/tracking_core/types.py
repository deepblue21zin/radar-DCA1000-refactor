from dataclasses import dataclass
from enum import Enum, auto


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
    measurement_quality: float = 1.0
    measurement_residual_m: float = 0.0
    is_primary: bool = False


@dataclass
class Track:
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
    last_measurement_quality: float = 1.0
    last_measurement_residual_m: float = 0.0

